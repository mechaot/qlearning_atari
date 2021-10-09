import numpy as np

#torch: shut up about indexing with uint8
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

from dqn_agent import DQNAgent
from utils import make_env, plot_learning_curve
from tqdm import tqdm
from time import perf_counter

import gym
#import ale_py
if __name__ == '__main__':
    #env_name = 'ALE/Tetris-v5'
    #env_name = 'Pong-v1'
    env_name = 'PongNoFrameskip-v4'
    env = make_env(env_name)
    best_score = -np.inf
    load_checkpoint = False
    render_episode = False

    n_games = 2000
    agent = DQNAgent(
        gamma=0.99, epsilon=1.0, eps_min=.1, eps_dec=2e-6,
        lr=1e-4,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=80000,
        batch_size=64,
        replace=1000,
        chkpoint_dir='./checkpoints',
        env_name=env_name
    )

    if load_checkpoint:
        agent.load_models()

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

                if render_episode:
                    fname = "frame_e{i:05}_f{i:05}.png".format(episode, step)

            _toc = perf_counter()
            scores.append(score)
            steps.append(n_steps)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])
            t = _toc-_tic
            print("episode: {i}, score: {score:.2f}, avg: {avg_score:.2f}, best: {best_score:.2f}, e: {agent.epsilon}, steps: {n_steps}, t: {t:.2f}".format(**locals()))

            if avg_score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score
    except KeyboardInterrupt:
        print("Canceled by user request")

    plot_learning_curve(steps, scores, eps_history, figure_file)

    print("done.")
