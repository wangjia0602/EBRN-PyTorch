from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from model.rcan import ResidualGroup
from model.common import *


class brm(nn.Module):
    def __init__(self, feat, scale):
        super(brm, self).__init__()

        self.up = nn.Sequential(
                nn.ConvTranspose2d(feat, feat, scale, stride = scale, padding = 0),
                nn.PReLU()
                           )

        self.up_conv = nn.Sequential(*[nn.Conv2d(feat, feat, 3, stride = 1, padding = 1) for _ in range(3)])

        self.down = nn.Sequential(
                nn.Conv2d(feat, feat, scale, stride = scale, padding = 0),
                nn.PReLU()
                           )
        self.down_conv = nn.Sequential(*[nn.Conv2d(feat, feat, 3, stride = 1, padding = 1) for _ in range(3)])



    def forward(self, x):
        out = self.up(x)
        up = self.up_conv(out)
        down = x - self.down(out)
        out = self.down_conv(down)
        out += down

        return up, out



class EBRN(nn.Module):
    def __init__(self, args):
        super(EBRN, self).__init__()
        feat = args.n_feats
        scale = args.scale
        n_resgroups = args.n_resgroups

        self.head1 = nn.Sequential(*[nn.Conv2d(3, feat*4, 3, stride = 1, padding = 1)])

        self.head2 = nn.Sequential(*[nn.Conv2d(feat*4, feat, 3, stride = 1, padding = 1)])
        self.head3 = nn.Sequential(*[nn.Conv2d(feat, feat, 3, stride = 1, padding = 1) ])


        self.brm = nn.ModuleList([brm(feat=feat, scale=scale) for _ in range(n_resgroups)])
        # self.brm_last = brm(feat=feat, scale=scale)

        self.conv = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride = 1, padding = 1)for _ in range(n_resgroups-1)])

        self.tail = nn.Sequential(*[nn.Conv2d(n_resgroups*feat, 3, 3, stride = 1, padding = 1)])



    def forward(self, x):
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
        for i, conv in enumerate(self.conv):
            x2 = up[-i] + up[-i + 1]
            x2 = conv(x2)
            out.append(x2)
        out = torch.cat(out, dim=1)
        out = self.tail(out)

        return out


