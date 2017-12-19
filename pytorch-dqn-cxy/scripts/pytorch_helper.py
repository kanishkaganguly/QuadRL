#! /usr/bin/env python3
import math
import os
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input, hidden, output):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.linear1 = nn.Linear(input, hidden)

        # self.bn1 = nn.BatchNorm2d(hidden)
        # self.linear2 = nn.Linear(hidden,hidden)
        # self.bn2 = nn.BatchNorm2d(hidden)

        self.linear3 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x).view(-1, self.hidden)
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = self.linear2(x)
        # x = self.bn2(x)

        x = F.relu(x)
        x = self.linear3(x)

        return x


class QuadDQN(object):
    def __init__(self, cuda, epoch_size, episode_size):
        self.cuda = cuda
        self.epoch_size = epoch_size
        self.episode_size = episode_size
        self.input = 4
        self.action = 8
        self.hidden = 32
        self.memory = ReplayMemory(10000)
        self.batch_size = 512
        if self.cuda:
            self.x = Variable(torch.randn(1, self.input)).cuda()
            self.y = Variable(torch.randn(1, self.action), requires_grad=False).cuda()
            self.model = torch.nn.Sequential(torch.nn.Linear(self.input, self.hidden), torch.nn.ReLU(),
                                             torch.nn.Linear(self.hidden, self.action)).cuda()
            self.loss_fn = torch.nn.MSELoss(size_average=True).cuda()
        else:
            self.x = Variable(torch.randn(1, self.input))
            self.y = Variable(torch.randn(1, self.action), requires_grad=False)
            self.model = DQN(self.input, self.hidden, self.action)
            self.loss_fn = torch.nn.MSELoss(size_average=True)

        self.learning_rate = .01
        self.eps = 0.1
        self.eps_list = np.linspace(self.eps, 1.0, self.epoch_size)
        self.gamma = 0.9
        self.gamma_list = np.linspace(self.gamma, 1.0, self.epoch_size)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optim, step_size=self.episode_size, gamma=0.1)
        self.loss = 0.0

    # Predict next action
    def predict_action(self, state):
        if self.cuda:
            self.x = torch.from_numpy(state).cuda()
            self.y = self.model(Variable(self.x)).cuda()
            return self.y.data.cpu().numpy()
        else:
            self.x = torch.from_numpy(state)
            self.x.view(-1, self.input)
            self.y = self.model(Variable(self.x))
            return self.y.data.numpy()

    def DQN_update(self):
        if len(self.memory) < 10:
            return
        elif len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        self.model.train()

        ByteTensor = torch.cuda.ByteTensor if self.cuda else torch.ByteTensor

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]).view(-1, self.input),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state).view(-1, self.input))
        action_batch = Variable(torch.from_numpy(np.asarray(batch.action)).view(-1, 1))
        reward_batch = Variable(torch.from_numpy(np.asarray(batch.reward)).float().view(-1, 1))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(batch_size).type(Tensor), requires_grad=False)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.view(batch_size, -1) * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    # Do backprop
    def backprop(self):
        print("Learning Rate: %f" % self.scheduler.get_lr()[0])
        self.optim.zero_grad()
        self.loss.backward()
        self.scheduler.step()

    # Get reward

    def get_reward2(self, curr_relative_state, new_relative_state):

        reward = np.linalg.norm(curr_relative_state) - np.linalg.norm(new_relative_state)
        print("Reward: %f" % reward)
        return reward

    def get_reward(self, curr_state, target_state):
        deviation_x = np.linalg.norm(curr_state[0] - target_state[0])
        deviation_y = np.linalg.norm(curr_state[1] - target_state[1])
        deviation_z = np.linalg.norm(curr_state[2] - target_state[2])
        deviation_yaw = np.linalg.norm(curr_state[3] - target_state[3])

        sigma_x = 0.1
        sigma_y = 0.1
        sigma_z = 0.01
        sigma_yaw = 0.1
        reward_x = math.exp(-deviation_x ** 2 / (2 * sigma_x))
        reward_y = math.exp(-deviation_y ** 2 / (2 * sigma_y))
        reward_z = math.exp(-deviation_z ** 2 / (2 * sigma_z))
        reward_yaw = math.exp(-deviation_yaw ** 2 / (2 * sigma_yaw))

        reward = self.sigmoid(0.9 * reward_x + 0.9 * reward_y + 0.9 * reward_z + 0.1 * reward_yaw)
        print("Reward: %f" % reward)
        return reward

    # Sigmoid
    def sigmoid(self, val):
        return math.exp(val) / (math.exp(val) + 1)

    # Value to action converter
    def convert_action(self, action):
        converter = {
            0: 'FWD',
            1: 'BCK',
            2: 'LFT',
            3: 'RGT',
            4: 'UP',
            5: 'DWN',
            6: 'ROT_CW',
            7: 'ROT_CCW'
        }
        return converter.get(action)

    # Do action
    def do_action(self, action_val):
        return self.convert_action(action_val)

    # Save weights
    def save_wts(self, savefile, epoch):
        saveme = {
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optim.state_dict(),
            'epoch'     : epoch,
            'epsilon'   : self.eps,
            'gamma'     : self.gamma
        }
        torch.save(saveme, savefile)

    # Load weights
    def load_wts(self, savefile):
        print("Loading saved model: %s\n" % savefile)
        if os.path.isfile(savefile):
            checkpoint = torch.load(savefile)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.eps = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
        else:
            return 0
