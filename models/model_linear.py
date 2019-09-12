import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel


class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        self.upsample = nn.Upsample(scale_factor=self.config.scale_factor, mode='linear')


    def forward(self, x):
        out = self.upsample(x)

        return out