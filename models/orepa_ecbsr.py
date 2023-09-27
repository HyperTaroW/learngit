from models.orepa_ecb import OREPA_ECB
import torch.nn as nn
import torch

class OREPA_ECBSR(nn.Module):
    def __init__(self, module_nums, channels_nums, act_type, scale, colors):
        super(OREPA_ECBSR, self).__init__()

        self.module_nums = module_nums
        self.channel_nums = channels_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        # self.deploy = deploy

        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [OREPA_ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type,
                               )]
        for i in range(self.module_nums):
            backbone += [OREPA_ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0,
                                   act_type=self.act_type)]
        backbone += [OREPA_ECB(self.channel_nums, self.colors*self.scale*self.scale,
                               depth_multiplier=2.0, act_type='linear')]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        y = self.backbone(x) + x
        y = self.upsampler(y)
        return y
