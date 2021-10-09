import collections
import cv2
from os import makedirs
from os.path import dirname, normpath, isdir, abspath

import numpy as np
import matplotlib.pyplot as plt
import gym
#import ale_py
#from ale_py import roms


def plot_learning_curve(x, scores, epsilons, filename):
    '''
        matplotlib scores and  epsilon over episodes
    '''
    f = plt.figure()
    ax = f.add_subplot(111, label="1")
    ax2 = f.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', color="C0")
    ax.tick_params(axis='y', color="C0")

    N = len(scores)
    running_av = np.empty(N)
    for t in range(N):
        running_av[t] = np.mean(scores[max(0, t-100): t+1])

    ax2.scatter(x, running_av, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")

    if not isdir(dirname(filename)):
        makedirs(dirname(filename))
    plt.savefig(filename)
    print("Saved {}".format(normpath(abspath(filename))))


class RepeatActionAndMaxFrame(gym.Wrapper):
    '''
        wrap the original environment to the following traits:

        execute a selected action for up to n frames (unless episode finishes)
        
        combine 2 subsequent image frames because Atari will show some
        sprites/objects only in even/odd frames because of sprite-buffer hw-limits
    '''
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)  # how we 'wrap'
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        '''
            execute action 'repeat' times into the original env

            combine output frames from original env to give 
            the "full" observation with all sprites
        '''
        t_reward = 0.0   # aggregate step rewards
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward, -1, 0)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs  # save even and odd frames
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self, **kwargs):
        '''
            reset game to start new episode
        '''
        obs = self.env.reset(**kwargs)                       # get initial state
        no_op_count = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_op_count): # do a few step of nothing at the beginning
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first: # force agent to fire after initial no_ops
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)
        self.frame_buffer = np.zeros_like((2, self.shape))   # reset framebuffer
        self.frame_buffer[0] = obs                           # load first obs in fb
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    '''
        preprocess single environment frames

        to grayscale
        scale down
        roll channel axis (numpy (h,w,c) -> torch(c,h,w))
    '''
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], *shape[:2])
        self.observation_space = gym.spaces.Box(
                                    low=0.0, high=1.0, # pixel value range to expect
                                    shape=self.shape,  # obervation dimensions (frame pixels)
                                    dtype=np.float32)

    def observation(self, obs):
        grayscale_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(grayscale_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        # new_obs = np.array(resized_screen, dtype=np.dtype=uint8).reshape(self.shape)
        # new_obs = new_obs / 255
        new_obs = resized_screen.reshape(self.shape) * (1. / 255)
        return new_obs


class StackFrames(gym.ObservationWrapper):
    """
        stack n frames, so we have a "small history" 
        or "sense of motion"
        (e.g. on pong the direction of travel of the ball l2r or r2l)
    """
    def __init__(self, env, count):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                        env.observation_space.low.repeat(count, axis=0),
                        env.observation_space.high.repeat(count, axis=0),
                        dtype=np.float32)
        self.stack = collections.deque(maxlen=count)  # init as ring buffer

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape) # high.shape should be the same

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, 
            clip_rewards=False, no_ops=0, fire_first=False, # for testing
            **kwargs):
    '''
        create the gym env and apply our wrappers
    '''
    env = gym.make(env_name, **kwargs)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, count=repeat)
    return env
