import torch.nn as nn
import torch.nn.functional as F

#as defined by https://arxiv.org/pdf/1511.00561.pdf (segnet paper)
class SegNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.batchMomentum = 0.1
        #classification / encoder from https://arxiv.org/pdf/1409.1556.pdf (VGG16 classification network)
        #multiple configurations presented in this paper, but segnet paper denotes 13 layers used,
        #of which there is only one such configuration
        #TODO: image however has 16 layers. as does https://github.com/delta-onera/segnet_pytorch/blob/master/segnet.py
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

        self.conv30 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch30 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch31 = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        self.conv40 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch40 = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch41 = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        #decoder
        self.conv41d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch41d = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv40d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch40d = nn.BatchNorm2d(512, momentum=self.batchMomentum)

        self.conv31d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch31d = nn.BatchNorm2d(512, momentum=self.batchMomentum)
        self.conv310d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.batch30d = nn.BatchNorm2d(256, momentum=self.batchMomentum)

        self.conv21d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch21d = nn.BatchNorm2d(256, momentum=self.batchMomentum)
        self.conv210d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
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
        #encoding layer
        x00 = F.relu(self.batch00(self.conv00(x)))
        x01 = F.relu(self.batch01(self.conv01(x00)))
        x0Pool, xIdx0 = F.max_pool2d(x01, kernel_size=2, stride=2, return_indices=True)

        x10 = F.relu(self.batch10(self.conv10(x0Pool)))
        x11 = F.relu(self.batch11(self.conv11(x10)))
        x1Pool, xIdx1 = F.max_pool2d(x11, kernel_size=2, stride=2, return_indices=True)

        x20 = F.relu(self.batch20(self.conv20(x1Pool)))
        x21 = F.relu(self.batch21(self.conv21(x20)))
        x2Pool, xIdx2 = F.max_pool2d(x21, kernel_size=2, stride=2, return_indices=True)

        x30 = F.relu(self.batch30(self.conv30(x2Pool)))
        x31 = F.relu(self.batch31(self.conv31(x30)))
        x3Pool, xIdx3 = F.max_pool2d(x31, kernel_size=2, stride=2, return_indices=True)

        x40 = F.relu(self.batch40(self.conv40(x3Pool)))
        x41 = F.relu(self.batch41(self.conv41(x40)))
        x4Pool, xIdx4 = F.max_pool2d(x41, kernel_size=2, stride=2, return_indices=True)

        #decoding layer
        x4Unpool = F.max_unpool2d(x4Pool, xIdx4, kernel_size=2, stride=2)
        x41 = F.relu(self.batch41d(self.conv41d(x4Unpool)))
        x40 = F.relu(self.batch40d(self.conv40d(x41)))

        x3Unpool = F.max_unpool2d(x40, xIdx3, kernel_size=2, stride=2)
        x31 = F.relu(self.batch31d(self.conv31d(x3Unpool)))
        x30 = F.relu(self.batch30d(self.conv30d(x31)))

        x2Unpool = F.max_unpool2d(x30, xIdx2, kernel_size=2, stride=2)
        x21 = F.relu(self.batch21d(self.conv21d(x2Unpool)))
        x20 = F.relu(self.batch20d(self.conv20d(x21)))

        x1Unpool = F.max_unpool2d(x20, xIdx1, kernel_size=2, stride=2)
        x11 = F.relu(self.batch11d(self.conv11d(x1Unpool)))
        x10 = F.relu(self.batch10d(self.conv10d(x11)))

        x0Unpool = F.max_unpool2d(x10, xIdx0, kernel_size=2, stride=2)
        x01 = F.relu(self.batch01d(self.conv01d(x0Unpool)))
        # return F.relu(self.batch00d(self.conv00d(x01)))
        return F.relu(self.conv00d(x01))
