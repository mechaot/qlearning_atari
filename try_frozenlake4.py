import gym
import numpy as np
import matplotlib.pyplot as plt
import random

class Agent():
    '''
        an "epsilon-greedy" tabular policy agent
    '''
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_decay):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_decay

        self.Q = {} # Tabular

        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                #if state >= self.n_states-1:
                    self.Q[(state, action)] = 0.0        # essential for finish to be zero, others are ok to be zero
                #else:
                #    self.Q[(state, action)] = -np.random.random()

    def greedy_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
        action = np.argmax(actions) # note: we should random choose if multiple max values
        return action

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            action = np.argmax(actions) # note: we should random choose if multiple max values
        return action

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

    def learn(self, state, action, reward, state_):
        #a_max = self.greedy_action(state_)
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.lr*(reward + 
                                            self.gamma*self.Q[(state_, a_max)] -
                                            self.Q[(state, action)])
        self.decrement_epsilon()



env = gym.make('FrozenLake-v1', map_name="4x4")
#from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
#env = FrozenLakeEnv(map_name="4x4", is_slippery=False)


agent = Agent(lr=0.002, gamma=0.93, eps_start=1.0, eps_end=0.01, eps_decay=0.9999995, n_actions=4, n_states=16)

scores = []
win_pct_list = []
n_games = 500000

#Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
#States: coord, ravelled

for r in range(n_games):
    observation = env.reset() # observation = state
    done = False
    score = 0
    while not done:
        a = agent.choose_action(observation)
        observation_, reward, done, info = env.step(a) # take a random action
        agent.learn(observation, a, reward, observation_)

        observation = observation_
        score += reward
        #env.render()
    
    scores.append(score)
    if r % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
        if r % 1000 == 0:
            print(f'episode: {r}/{n_games}, win %: {win_pct:.3f}, e={agent.epsilon:.4f}' )


plt.plot(win_pct_list)
plt.show()


env.close()
