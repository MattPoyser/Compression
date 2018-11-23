import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.module):
    #TODO see Alexnet
    # def __init__(self, inputChannels, transform=None):
    #     self.input_channels = inputChannels
    #     self.transform = transform
    #
    #     #5 conv layers. TODO: 4096 output but no idea what the hidden layers / kernels / padding are??
    #     self.conv0 = nn.Conv2d(self.input_channels, 96, kernel_size=11, stride=4)
    #     self.conv1 = nn.Conv2d(96, 256, kernel_size=5, padding=1)
    #     self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(2048, 4096, kernel_size=3, padding=1)
    #
    #     #2 fully connected layers
    #     self.fcn0 = nn.Linear(4096, 4096)
    #     self.fcn1 = nn.Linear(4096, 4096)
    #
    # def forward(self, x):
    #     x0 = F.relu(self.conv0(x))
    #     x0Pool = F.max_pool2d(x0, )
