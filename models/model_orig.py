import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel


class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        d = 56 # out channels of first layer
        s = 12 # out channels of hidden layer
        m = 4 # number of layer of hidden layer block

        # Feature extraction
        self.first_part = ConvBlock(self.config.num_channels, d, 5, 1, 0, activation='prelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(d, s, 1, 1, 0, activation='prelu', norm=None))
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(ResnetBlock(s, 3, 1, 1, activation='prelu', norm='batch'))
        self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(s, d, 1, 1, 0, activation='prelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose1d(d, self.config.num_channels, 25, self.config.scale_factor, 0, output_padding=0)
        # self.last_part = torch.nn.Sequential(
        #     Upsample2xBlock(d, d, upsample='rnc', activation=None, norm=None),
        #     Upsample2xBlock(d, num_channels, upsample='rnc', activation=None, norm=None)
        # )

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()




