import numpy as np
from os.path import join, dirname
from os import makedirs
#torch: shut up about indexing with uint8
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

#from dqn_agent import DQNAgent as Agent
#from ddqn_agent import DoubleDQNAgent as Agent
from duelingqn_agent import DuelingDQNAgent as Agent
#from dueling_double_dqn_agent import DuelingDoubleDQNAgent as Agent

from utils import make_env, plot_learning_curve
from tqdm import tqdm
from time import perf_counter
from imageio import imwrite
import torch

import gym
from gym import wrappers

if __name__ == '__main__':
    #env_name = 'ALE/Tetris-v5'
    #env_name = 'Pong-v1'
    print("Cuda:", torch.cuda.is_available(), torch.cuda.current_device())

    env_name = 'PongNoFrameskip-v4'
    env = make_env(env_name)
    best_score = -np.inf
    load_checkpoint = False
    render_episode = 50 # render every nth episode

    n_games = 300 # 500, 2000
    agent = Agent(
        gamma=0.99, epsilon=1.0, eps_min=.1, eps_dec=2e-6,
        lr=1e-4,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=80000,
        batch_size=64,
        replace=1000,
        chkpoint_dir='./checkpoints_dqna',
        env_name=env_name
    )

    if load_checkpoint:
        agent.load_models(include_memory=True)

    #render episode
    # makedirs(`./video`, exist_ok=True)
    # env = wrappers.Monitor(env, './video', video_callable=lambda episode_id: True, force=True)

    #figure_file = f"{agent.checkpoint_dir}/{agent.algo}_{env_name}_lr{agent.lr}_{n_games}.png"
    figure_file = agent.checkpoint_dir+"/"+agent.algo+"_"+env_name+"_lr"+str(agent.lr)+"_"+str(n_games)+".png"

    n_steps = 0
    scores, eps_history, steps = [], [], []

    try:
        for episode in range(n_games):
            _tic = perf_counter()
            done = False
            score = 0
            step = 0
            obs = env.reset()

            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                score += reward

                if not load_checkpoint:
                    agent.store_transition(obs, action, reward, obs_, int(done))
                    agent.learn()

                obs = obs_
                n_steps += 1
                step += 1

                if episode % render_episode == 0:# and render_episode > 0:
                    fname = join(agent.checkpoint_dir, "e{:05d}/frame{:07d}.png".format(episode, step))
                    makedirs(dirname(fname), exist_ok=True)
                    imwrite(fname, (obs[-1,...] * 255).astype(np.uint8))
                    #  ffmpeg -r 30 -i frame%07d.png -c:v libvpx-vp9 -crf 20 -b:v 2M -y out.mp4

            _toc = perf_counter()
            scores.append(score)
            steps.append(n_steps)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])
            t = _toc-_tic
            print("episode: {episode}, score: {score:.2f}, avg: {avg_score:.2f}, best: {best_score:.2f}, e: {agent.epsilon}, steps: {n_steps}, t: {t:.2f}".format(**locals()))

            if avg_score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score
    except KeyboardInterrupt:
        print("Canceled by user request. Saving state")

    plot_learning_curve(steps, scores, eps_history, figure_file)
    
    print("Saving models.")
    agent.save_models(include_memory=True)
    print("done.")
