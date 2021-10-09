import numpy as np
import torch as T

from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

#pylint: disable=no-member

class DQNAgent():
    def __init__(self, 
            env_name, input_shape, n_actions,
            mem_size=50000, replace=1000, batch_size=32,
            epsilon=0.9, eps_min=.01, eps_dec=5e-7,
            gamma=0.99, lr=.99, algo='DQNAgent', chkpoint_dir="./checkpoint"):
        self.env_name = env_name           # name of game and gym env
        self.input_shape = input_shape       # image shape
        self.n_actions = n_actions         # number of possible actions
        self.replace_target_cnt = replace  # how often to update the 2nd CNN the target one (learning steps)
        self.algo = algo                   # algorithm name 
        self.batch_size = batch_size       # memory replay batch size
        self.epsilon = epsilon             # greedy-ness
        self.eps_min = eps_min             # minimum exploration/max greedyeness
        self.eps_dec = eps_dec             # epsilon decay
        self.gamma = gamma                 # future step discount factor 
        self.lr = lr                       # learning rate
        self.checkpoint_dir = chkpoint_dir # were to save checkpoints
        
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0 # when to update target network from eval network

        self.memory = ReplayBuffer(mem_size, self.input_shape, self.n_actions)

        #this will have the real learning and back-propagation
        self.q_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, input_shape=self.input_shape,
                                name=self.env_name+"_"+self.algo+'_q_eval',
                                checkpoint_dir=self.checkpoint_dir)

        #this will get the parameters copied to occationally
        self.q_next = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, input_shape=self.input_shape,
                                name=self.env_name+"_"+self.algo+'_q_next',
                                checkpoint_dir=self.checkpoint_dir)


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # be greedy
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        stateT = T.tensor(state).to(self.q_eval.device)
        actionT = T.tensor(action).to(self.q_eval.device)
        rewardT = T.tensor(reward).to(self.q_eval.device)
        state_T = T.tensor(state_).to(self.q_eval.device)
        doneT = T.Tensor(done).type(T.bool).to(self.q_eval.device)

        return stateT, actionT, rewardT, state_T, doneT
        
    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            print("Syncing networks")

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        # if not enough samples in memory, do not learn, keep dumb and try ;)
        if self.memory.mem_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.update_target_network()

        # replac memory actions
        states, actions, rewards, states_, dones = self.sample_memory()

        # action values for batch of states
        # slice out the values for actions we remember to have performed
        indices = np.arange(self.batch_size, dtype=int)
        q_pred = self.q_eval.forward(states)[indices, actions]
        #q_pred = self.q_eval.forward(states)[:, actions]
        q_next, _ = self.q_next.forward(states_).max(dim=1)

        q_next[dones] = 0.0  # where terminal, value is zero

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()
