from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from model.rcan import ResidualGroup
from model.common import *
from model import common

class brm(nn.Module):
    def __init__(self, feat, scale):
        super(brm, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(feat, feat, scale, stride=scale, padding=0),
            nn.PReLU()
        )
        self.up_conv = nn.ModuleList()
        for _ in range(3):
            self.up_conv.append(nn.Conv2d(feat, feat, 3, stride=1, padding=1))
            self.up_conv.append(nn.PReLU())

        self.down = nn.Sequential(
            nn.Conv2d(feat, feat, scale, stride=scale, padding=0),
            nn.PReLU()
        )
        # self.down_conv = nn.Sequential(*[nn.Conv2d(feat, feat, 3, stride = 1, padding = 1) for _ in range(3)])
        self.down_conv = nn.ModuleList()
        for _ in range(3):
            self.down_conv.append(nn.Conv2d(feat, feat, 3, stride=1, padding=1))
            self.down_conv.append(nn.PReLU())

    def forward(self, x):
        up_out = self.up(x)
        # up_out = out
        up = up_out.clone()
        for conv in self.up_conv:
            up = conv(up)

        out = x - self.down(up_out.clone())

        down = out.clone()
        for conv in self.down_conv:
            down = conv(down)

        out += down

        return up, out


class EBRN(nn.Module):
    def __init__(self, args):
        super(EBRN, self).__init__()
        feat = args.n_feats
        scale = args.scale
        self.n_resgroups = args.n_resgroups

        self.head1 = nn.Sequential(nn.Conv2d(3, feat * 4, 3, stride=1, padding=1), nn.PReLU())

        self.head2 = nn.Sequential(nn.Conv2d(feat * 4, feat, 3, stride=1, padding=1), nn.PReLU())
        self.head3 = nn.Sequential(nn.Conv2d(feat, feat, 3, stride=1, padding=1), nn.PReLU())

        self.brm = nn.ModuleList([brm(feat=feat, scale=scale) for _ in range(self.n_resgroups)])
        # self.brm_last = brm(feat=feat, scale=scale)

        self.conv = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride=1, padding=1) for _ in range(self.n_resgroups - 1)])
        self.relu = nn.ModuleList([nn.PReLU() for _ in range(self.n_resgroups - 1)])

        self.tail = nn.Sequential(nn.Conv2d(self.n_resgroups * feat, 3, 3, stride=1, padding=1), nn.PReLU())
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)

        out = self.head1(x)
        out = self.head2(out)
        out = self.head3(out)
        up = []
        # down = []
        x2 = out
        for unit in self.brm:
            x1, x2 = unit(x2)
            up.append(x1)

            # down.append(x2)
        out = []
        out.append(up[-1])
        for i, conv, relu in zip(range(self.n_resgroups - 1), self.conv, self.relu):
            if i ==0:
                x2 = up[-1] + up[-2]
            else:
                x2 += up[-i-2]
            x2 = conv(x2)
            x2 = relu(x2)
            out.append(x2)
        out = torch.cat(out, dim=1)
        out = self.tail(out)
        out = self.add_mean(out)



        return out



