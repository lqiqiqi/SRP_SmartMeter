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

        # self.upsample = nn.Upsample(scale_factor=self.config.scale_factor, mode='nearest')
        if self.config.scale_factor == 10:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        elif self.config.scale_factor == 100:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        elif self.config.scale_factor == 1000:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        else:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.scale_factor, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=self.config.scale_factor),
                nn.LeakyReLU(0.2)
            ])

        # Feature extraction
        # if the last is conv, padding is 2
        self.first_part = ConvBlock(self.config.num_channels, self.config.d, self.config.k, 1, 0, activation='lrelu', norm=None)

        self.layers = []
        # Shrinking
        self.layers.append(ConvBlock(self.config.d, self.config.s, 1, 1, 0, activation='lrelu', norm=None))
        # Non-linear Mapping
        for _ in range(int(self.config.m)):
            self.layers.append(ResnetBlock(self.config.s, 3, 1, 1, activation='lrelu', norm='instance'))
        self.layers.append(nn.PReLU())
        # Expanding
        self.layers.append(ConvBlock(self.config.s, self.config.d, 1, 1, 0, activation='lrelu', norm=None))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose1d(int(self.config.d), int(self.config.num_channels), int(self.config.k), 1, 0, output_padding=0)
        # self.last_part = ConvBlock(self.config.d, self.config.num_channels, 5, 1, 2, activation=None, norm=None)

        # self.last_part = torch.nn.Sequential(
        #     Upsample2xBlock(d, d, upsample='rnc', activation=None, norm=None),
        #     Upsample2xBlock(d, num_channels, upsample='rnc', activation=None, norm=None)
        # )

    def forward(self, x):
        out = self.upsample(x)
        out = self.first_part(out)
        residual = out
        out = self.mid_part(out)
        out = torch.add(out, residual)
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


class SRPShuffle(nn.Module):
    def __init__(self, config):
        super(SRPShuffle, self).__init__()

    def forward(self, input):
        B, C, L = input.size()  # 假设1，10，10
        if C % self.config.scale != 0:
            raise Exception('Check input channels')
        out_L = self.config.scale_factor * L  # out_L = 100
        out_C = C // self.config.scale_factor  # out_C = 1

        input_view = input.contiguous().view((B, out_C, self.config.scale_factor, L)) # 1， 1， 10， 10
        view_permu = input_view.permute(0, 1, 3, 2).contiguous() # 1， 1， 10， 10
        return view_permu.view((B, out_C, out_L)) # 1， 1， 100

    def __repr__(self):
        return self.__class__.__name__ + '(sequence_upscale_factor=' + str(self.upscale_factor) + ')'


class SRPUpsampleBlock(nn.Module):
    def __init__(self, scale=10, activation=nn.functional.relu):
        super(SRPUpsampleBlock, self).__init__()
        self.act = activation
        self.shuffler = SRPShuffle(scale)

    def forward(self, x):
        return self.act(self.shuffler(x))

