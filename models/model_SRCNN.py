import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.first_layer = nn.Conv1d(in_channels=1, out_channels=256,
                                     kernel_size=9, stride=1, padding=4, bias=False)

        m = 7
        self.layers = []
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False))
            self.layers.append(nn.PReLU())
        self.mid_part = nn.Sequential(*self.layers)

        # Deconvolution
        # TODO stride, padding, outpadding需要根据scale修改 n_out=(n_in−1)×S−2P+F
        self.last_part = nn.ConvTranspose1d(256, 1, kernel_size=7, stride=10, padding=1,
                                            output_padding=1)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out


