import torch
import torch.nn as nn
from base.base_networks import *
from base.base_model import BaseModel


class Net(torch.nn.Module, BaseModel):
    def __init__(self, config):
        super(Net, self).__init__() # super只能init第一个父类
        BaseModel.__init__(self, config)

        L = 4
        n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        self.downsampling_l = []
        self.down_layers = []

        # downsampling layers
        for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
            self.temp_layers = []
            self.temp_layers.append(nn.Conv1d(in_channels=self.config.num_channels, out_channels=nf, kernel_size=fs,
                      stride=2, padding=nf//2, bias=False))
            # if l > 0: x = BatchNormalization(mode=2)(x)
            self.temp_layers.append(nn.LeakyReLU(0.2))
            self.down_layers.append(torch.nn.Sequential(*self.temp_layers))

        # bottleneck layer
        self.bottleneck = []
        self.bottleneck.append(nn.Conv1d(in_channels=n_filters[-1], out_channels=n_filters[-1], kernel_size=n_filtersizes[-1],
                                    stride=2, padding=n_filters[-1] // 2, bias=False))
        self.bottleneck.append(nn.Dropout(p=0.5))
        # x = BatchNormalization(mode=2)(x)
        self.bottleneck.append(nn.LeakyReLU(0.2))
        self.mid_layer = torch.nn.Sequential(*self.bottleneck)

        # upsampling layers
        self.up_layers = []
        for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
            # (-1, n/2, 2f)
            self.temp_layers = []
            self.temp_layers.append(nn.Conv1d(in_channels=n_filters[-1], out_channels=2*nf, kernel_size=fs,
                                    padding=fs // 2, bias=False))
            # x = BatchNormalization(mode=2)(x)
            self.temp_layers.append(nn.Dropout(p=0.5))
            self.temp_layers.append(nn.ReLU())
            # (-1, n, f)
            self.temp_layers.append(nn.PixleShuffle(r=2))
            # (-1, n, 2f)
            self.up_layers.append(torch.nn.Sequential(*self.temp_layers))

        self.last_layer = []
        self.last_layer.append(nn.Conv1d(in_channels=1, out_channels=2, kernel_size=9,
                                    padding=9 // 2, bias=False))
        self.last_layer.append(nn.PixleShuffle(r=2))

    def forward(self, x):
        out = self.down_layers[0](x)
        self.downsampling_l.append(out)
        out = self.down_layers[1](out)
        self.downsampling_l.append(out)
        out = self.down_layers[2](out)
        self.downsampling_l.append(out)
        out = self.down_layers[3](out)
        self.downsampling_l.append(out)

        out = self.mid_layer(out)

        out = self.up_layers[0](out)
        out = torch.cat((out, self.downsampling_l[-1]), -1)
        out = self.up_layers[1](out)
        out = torch.cat((out, self.downsampling_l[-2]), -1)
        out = self.up_layers[2](out)
        out = torch.cat((out, self.downsampling_l[-3]), -1)
        out = self.up_layers[3](out)
        out = torch.cat((out, self.downsampling_l[-4]), -1)

        out = self.last_layer(out)
        out = torch.add(out, x)

        return out

    def weight_init(self):
        print('weight is xavier initilized')
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv1d):
                # m.weight.data.normal_(mean, std)
                m.weight.data = nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()