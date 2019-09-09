import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from base.base_model import BaseModel


class SRPResNet_Residual_Block(nn.Module):
    def __init__(self, ndf=64):
        super(SRPResNet_Residual_Block, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=ndf, out_channels=ndf, kernel_size=5, stride=1, padding=2, bias=False)
        self.in1 = nn.InstanceNorm1d(ndf, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(in_channels=ndf, out_channels=ndf, kernel_size=5, stride=1, padding=2, bias=False)
        self.in2 = nn.InstanceNorm1d(ndf, affine=True, track_running_stats=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class SRPShuffle(nn.Module):
    def __init__(self, scale):
        super(SRPShuffle, self).__init__()
        self.scale = scale

    def forward(self, input):
        B, C, L = input.size()
        if C % self.scale != 0:
            raise Exception('Check input channels')
        out_L = self.scale * L # upsample
        out_C = C // self.scale # 10->1

        input_view = input.contiguous().view((B, out_C, self.scale, L))
        view_permu = input_view.permute(0, 1, 3, 2).contiguous()
        return view_permu.view((B, out_C, out_L)) # 把原来C = 10变成C = 1,L * 10起到upsample的效果

    def __repr__(self):
        return self.__class__.__name__ + '(sequence_upscale_factor=' + str(self.scale) + ')'


class SRPUpsampleBlock(nn.Module):
    def __init__(self, scale=10, activation=F.relu):
        super(SRPUpsampleBlock, self).__init__()
        self.act = activation
        self.shuffler = SRPShuffle(scale)

    def forward(self, x):
        return self.act(self.shuffler(x))


class Net(nn.Module, BaseModel):
    def __init__(self, config, residual_blocks=16, ndf=64):
        super(Net, self).__init__()
        BaseModel.__init__(self, config)
        self.ndf = ndf
        self.scale = self.config.scale_factor

        self.conv_input_1 = nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.conv_input_2 = nn.Conv1d(in_channels=ndf, out_channels=ndf, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu2 = nn.LeakyReLU()

        self.residual = self.make_layer(SRPResNet_Residual_Block, residual_blocks)

        self.conv_mid = nn.Conv1d(in_channels=ndf, out_channels=ndf, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn_mid = nn.InstanceNorm1d(ndf, affine=True, track_running_stats=True)

        if self.scale == 10:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        elif self.scale == 100:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        elif self.scale == 1000:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2),
                nn.Conv1d(in_channels=ndf, out_channels=ndf * 10, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=10),
                nn.LeakyReLU(0.2)
            ])
        else:
            self.upsample = nn.Sequential(*[
                nn.Conv1d(in_channels=ndf, out_channels=ndf * self.scale, kernel_size=5, stride=1, padding=2, bias=False),
                SRPUpsampleBlock(scale=self.scale),
                nn.LeakyReLU(0.2)
            ])

        self.conv_output = nn.Conv1d(in_channels=ndf, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)
        self.final_relu = nn.ReLU()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(ndf=self.ndf))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu2(self.conv_input_2(self.relu1(self.conv_input_1(x))))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upsample(out)
        out = self.final_relu(self.conv_output(out))
        return out.view((out.size()[0], 1, out.size()[2]))