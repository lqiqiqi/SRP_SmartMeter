import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(64, 64)
        self.c1 = ConvBlock(64 * 2, 64, 1, 1, 0, activation='prelu', norm=None)
        self.c2 = ConvBlock(64 * 3, 64, 1, 1, 0, activation='prelu', norm=None)
        self.c3 = ConvBlock(64 * 4, 64, 1, 1, 0, activation='prelu', norm=None)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        self.entry = nn.Conv1d(1, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ConvBlock(64 * 2, 64, 1, 1, 0, activation='prelu', norm=None)
        self.c2 = ConvBlock(64 * 3, 64, 1, 1, 0, activation='prelu', norm=None)
        self.c3 = ConvBlock(64 * 4, 64, 1, 1, 0, activation='prelu', norm=None)

        self.upsample = nn.ConvTranspose1d(64, 64, 10, self.config.scale_factor, 0, output_padding=0)

        self.exit = ConvBlock(64, 1, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):

        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3)

        out = self.exit(out)

        return out

    def weight_init(self):
        print('weight is xavier initilized')
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv1d):
                # m.weight.data.normal_(mean, std)
                m.weight.data = nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.0001)
                m.weight.data = nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()




