import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, lr=0.99, alpha=0.99, name="MyQNetwork", checkpoint_dir="./checkpoints"):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        self.checkpoint_dir = os.path.dirname(self.checkpoint_file)        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.input_shape = input_shape
        self.n_actions = n_actions

        input_channels = 1 if len(input_shape) == 2 else input_shape[0]

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, 
                                kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4,4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1)
        #fc_shape = np.prod(self.conv3.data.shape)
        fc_shape = self.get_conv_output_size(self.input_shape)
        self.fc1 = nn.Linear(fc_shape, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, alpha=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_conv_output_size(self, shape):
        '''
            pass a dummy size-1 batch trough the conv layers for retreival of its size
        '''
        dummy = torch.zeros(1, *shape)
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)
        return int(np.prod(dummy.size()))


    def forward(self, state):
        '''
            take an observation and return the action activations
        '''
        batch_size = state.size()[0] # x.size()[0]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # flatten(start_dims=1) shallow copy
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.softmax(x)
        return x

    def save_checkpoint(self):
        filename = self.checkpoint_file + ".cpt"
        torch.save(self.state_dict(), filename)
        print("Saved checkpoint {}".format(os.path.abspath(filename)))

    def load_checkpoint(self):
        filename = self.checkpoint_file + ".cpt"
        torch.load_state_dict(torch.load(filename))
        print("Loaded checkpoint {}".format(os.path.abspath(filename)))