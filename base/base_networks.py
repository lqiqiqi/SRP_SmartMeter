import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv1d(int(input_size), int(output_size), int(kernel_size), stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(int(output_size))
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(int(output_size))

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(int(num_filter), int(num_filter), int(kernel_size), stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv1d(int(num_filter), int(num_filter), int(kernel_size), stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(int(num_filter), affine=True)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(int(num_filter), affine=True)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, 1, 30000)


class EResidualBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels
                 ):
        super(EResidualBlock, self).__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_channels, out_channels, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_channels, out_channels, 1, 1, 0),
        )

        # self.init_weights()

    def forward(self, x):
        out = self.body(x)
        out = torch.nn.functional.relu(out + x)
        return out

    def init_weights(self):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, torch.nn.Conv1d):
                # m.weight.data.normal_(mean, std)
                m.weight.data = torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
