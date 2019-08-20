import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel


class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        # d = 56 # out channels of first layer       # s = 32 # out channels of hidden layer
        # self.config.m = 16 # number of layer of hidden layer block

        self.upsample = nn.Upsample(scale_factor=self.config.scale_factor, mode='bicubic')

        # Feature extraction
        self.first_part = ConvBlock(self.config.num_channels, self.config.d, 5, 1, 0, activation='prelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(self.config.d, self.config.s, 1, 1, 0, activation='prelu', norm=None))
        # Non-linear Mapping
        for _ in range(self.config.m):
            self.layers.append(ResnetBlock(self.config.s, 3, 1, 1, activation='prelu', norm='batch'))
        self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(self.config.s, self.config.d, 1, 1, 0, activation='prelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose1d(self.config.d, self.config.num_channels, 5, 1, 0, output_padding=0)
        # self.last_part = torch.nn.Sequential(
        #     Upsample2xBlock(d, d, upsample='rnc', activation=None, norm=None),
        #     Upsample2xBlock(d, num_channels, upsample='rnc', activation=None, norm=None)
        # )

    def forward(self, x):
        out = self.upsample(x)
        out = self.first_part(out)
        out = self.mid_part(out)
        out = self.last_part(out)
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




