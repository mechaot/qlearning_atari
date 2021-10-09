import numpy as np

class ReplayBuffer():
    '''
        get hold of 
    '''
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.input_shape = input_shape
        assert n_actions < 2**63-1
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        '''
            reset the memory bank completely
        '''
        self.state_mem = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.done_mem = np.zeros(self.mem_size, dtype=np.uint8)  # terminal_mem

    def store_transition(self, state, action, reward, state_, done):
        '''
            store action of what is kind like a ring buffer
        '''
        index = self.mem_counter % self.mem_size     # make it a ring-buffer
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = state_
        self.done_mem[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        '''
            get unique random samples from the memory bank

            batch_size: number of samples
        '''
        max_valid_idx = min(self.mem_counter, self.mem_size)
        #if batch_size >= max_valid_idx:
        #    raise ValueError("Not enough stored samples yet")
        batch = np.random.choice(max_valid_idx, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        dones = self.done_mem[batch]

        return states, actions, rewards, states_, dones