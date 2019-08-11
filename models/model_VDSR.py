import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel

class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        base_filter = 64
        num_residuals = 18

        self.convtransposed = nn.ConvTranspose1d(self.config.num_channels, self.config.num_channels, 10, self.config.scale_factor, 0, output_padding=0)
        self.input_conv = ConvBlock(self.config.num_channels, base_filter, 3, 1, 1, norm=None, bias=False)

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None, bias=False))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = ConvBlock(base_filter, self.config.num_channels, 3, 1, 1, activation=None, norm=None, bias=False)


    def forward(self, x):
        out = self.convtransposed(x)
        residual = out
        out = self.input_conv(out)
        out = self.residual_layers(out)
        out = self.output_conv(out)
        out = torch.add(out, residual)

        return out

    def weight_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Norm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
