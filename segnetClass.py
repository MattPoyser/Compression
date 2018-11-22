import torch.nn as nn
import torch.nn.functional as F


# as defined by https://arxiv.org/pdf/1511.00561.pdf (segnet paper)
class SegNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.batchMomentum = 0.1
        # classification / encoder from https://arxiv.org/pdf/1409.1556.pdf (VGG16 classification network)
        # multiple configurations presented in this paper, but segnet paper denotes 13 layers used,
        # of which there is only one such configuration
        # TODO: image however has 16 layers. as does https://github.com/delta-onera/segnet_pytorch/blob/master/segnet.py
        # TODO see VGGNET
        self.conv00 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.batch00 = nn.BatchNorm2d(64, momentum=self.batchMomentum)
        self.conv01 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch01 = nn.BatchNorm2d(64, momentum=self.batchMomentum)

        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(128, momentum=self.batchMomentum)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(128, momentum=self.batchMomentum)

        self.conv20 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch20 = nn.BatchNorm2d(256, momentum=self.batchMomentum)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch21 = nn.BatchNorm2d(256, momentum=self.batchMomentum)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch22 = nn.BatchNorm2d(256, momentum=self.batchMomentum)

        self.conv30 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch30 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch31 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch32 = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        self.conv40 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch40 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch41 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        # self.conv42 = nn.Conv2d(512, self.output_channels, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch42 = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        # convnet final layers
        self.fcc50 = nn.Linear(512, 4096)
        self.fcc51 = nn.Linear(4096, 4096)
        self.fcc52 = nn.Linear(4096, self.output_channels)

        # decoder
        self.conv41d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch41d = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv40d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch40d = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        self.conv31d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch31d = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv30d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.batch30d = nn.BatchNorm2d(256, momentum=self.batchMomentum)

        self.conv21d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch21d = nn.BatchNorm2d(256, momentum=self.batchMomentum)
        self.conv20d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batch20d = nn.BatchNorm2d(128, momentum=self.batchMomentum)

        self.conv11d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch11d = nn.BatchNorm2d(128, momentum=self.batchMomentum)
        self.conv10d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.batch10d = nn.BatchNorm2d(64, momentum=self.batchMomentum)

        self.conv01d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch01d = nn.BatchNorm2d(64, momentum=self.batchMomentum)
        self.conv00d = nn.Conv2d(64, self.output_channels, kernel_size=3, padding=1)
        # self.batch00d = nn.BatchNorm2d(self.output_channels, momentum=self.batchMomentum)

    def forward(self, x):
        # encoding layer
        x00 = F.relu(self.batch00(self.conv00(x)))
        x01 = F.relu(self.batch01(self.conv01(x00)))
        x0_pool, x_idx0 = F.max_pool2d(x01, kernel_size=2, stride=2, return_indices=True)

        x10 = F.relu(self.batch10(self.conv10(x0_pool)))
        x11 = F.relu(self.batch11(self.conv11(x10)))
        x1_pool, x_idx1 = F.max_pool2d(x11, kernel_size=2, stride=2, return_indices=True)

        x20 = F.relu(self.batch20(self.conv20(x1_pool)))
        x21 = F.relu(self.batch21(self.conv21(x20)))
        x22 = F.relu(self.batch22(self.conv22(x21)))
        x2_pool, x_idx2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        x30 = F.relu(self.batch30(self.conv30(x2_pool)))
        x31 = F.relu(self.batch31(self.conv31(x30)))
        x32 = F.relu(self.batch32(self.conv31(x31)))
        x3_pool, x_idx3 = F.max_pool2d(x32, kernel_size=2, stride=2, return_indices=True)

        x40 = F.relu(self.batch40(self.conv40(x3_pool)))
        x41 = F.relu(self.batch41(self.conv41(x40)))
        x42 = F.relu(self.batch42(self.conv42(x41)))
        # return F.relu(self.conv42(x41))
        x4_pool, x_idx4 = F.max_pool2d(x42, kernel_size=2, stride=2, return_indices=True)

        # convnet final layers
        # xView (not working) converts to a 1 x X array for some X tbc
        # xView = x4_pool.view(-1, 2*512*2*2*16)
        # x50 = F.relu(self.fcc50(xView))
        # x51 = F.relu(self.fcc51(x50))
        # x52 = F.relu(self.fcc52(x51))
        # return x52

        # decoding layer
        x4_unpool = F.max_unpool2d(x4_pool, x_idx4, kernel_size=2, stride=2)
        x41 = F.relu(self.batch41d(self.conv41d(x4_unpool)))
        x40 = F.relu(self.batch40d(self.conv40d(x41)))

        x3_unpool = F.max_unpool2d(x40, x_idx3, kernel_size=2, stride=2)
        x31 = F.relu(self.batch31d(self.conv31d(x3_unpool)))
        x30 = F.relu(self.batch30d(self.conv30d(x31)))

        x2_unpool = F.max_unpool2d(x30, x_idx2, kernel_size=2, stride=2)
        x21 = F.relu(self.batch21d(self.conv21d(x2_unpool)))
        x20 = F.relu(self.batch20d(self.conv20d(x21)))

        x1_unpool = F.max_unpool2d(x20, x_idx1, kernel_size=2, stride=2)
        x11 = F.relu(self.batch11d(self.conv11d(x1_unpool)))
        x10 = F.relu(self.batch10d(self.conv10d(x11)))

        x0_unpool = F.max_unpool2d(x10, x_idx0, kernel_size=2, stride=2)
        x01 = F.relu(self.batch01d(self.conv01d(x0_unpool)))
        # return F.relu(self.batch00d(self.conv00d(x01)))
        return F.relu(self.conv00d(x01))
