

import numpy as np
import torch
from torch import nn
from q_learning.utils import *
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_shp, out_number):
        super(DQN, self).__init__()
        h, w, c_in = input_shp
        #   Q-learning network as described in
        #   https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        #   out_number: number of action-value to output

        self.conv1 = nn.Conv2d(c_in, 32, kernel_size=8, stride=4)
        w_out = conv2d_size_out(w, kernel_size=8, stride=4)
        h_out = conv2d_size_out(h, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        w_out = conv2d_size_out(w_out, kernel_size=4, stride=2)
        h_out = conv2d_size_out(h_out, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        w_out = conv2d_size_out(w_out, kernel_size=3, stride=1)
        h_out = conv2d_size_out(h_out, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(w_out * h_out * 64, 512)
        self.head = nn.Linear(512, out_number)

    def forward(self, x):
        # correct the shape if needed to NCHW format
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DQN2(nn.Module):

    def __init__(self, input_shp, out_number):
        super(DQN, self).__init__()
        h, w, c_in = input_shp

        self.conv1 = nn.Conv2d(c_in, 16, kernel_size=5, stride=2)
        w_out = conv2d_size_out(w, kernel_size=5, stride=2)
        h_out = conv2d_size_out(h, kernel_size=5, stride=2)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        w_out = conv2d_size_out(w_out, kernel_size=5, stride=2)
        h_out = conv2d_size_out(h_out, kernel_size=5, stride=2)

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        w_out = conv2d_size_out(w_out, kernel_size=5, stride=2)
        h_out = conv2d_size_out(h_out, kernel_size=5, stride=2)

        self.bn3 = nn.BatchNorm2d(64)

        self.head = nn.Linear(w_out * h_out * 32, out_number)

    def forward(self, x):
        # correct the shape if needed to NCHW format
        # print(x.shape)#torch.Size([1, 64, 64, 3])
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)torch.Size([1, 3, 64, 64])

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class LSTMDQN(nn.Module):
    def __init__(self, n_action):
        super(LSTMDQN, self).__init__()
        self.n_action = n_action
        LSTM_MEMORY = 128

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=1, padding=1)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

        self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
        # self.affine2 = nn.Linear(2048, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x, hidden_state, cell_state):
        # CNN
        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state
