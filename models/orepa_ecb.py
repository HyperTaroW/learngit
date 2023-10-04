import torch
import torch.nn as nn
from models.ecb import SeqConv3x3
from models.convnet_utils import conv_bn, conv_bn_relu
import torch.nn.functional as F

class OREPA_1x1_3x3(nn.Module):
    def __init__(self, seq_type, in_planes, out_planes, kernel_size, depth_multiplier):
        super(OREPA_1x1_3x3, self).__init__()

        self.in_channels = in_planes
        self.out_channels = out_planes
        self.mid_channels = int(out_planes * depth_multiplier)
        self.type = seq_type

        self.conv1 = conv_bn(self.in_channels, self.mid_channels, kernel_size=1)
        self.conv2 = conv_bn(self.mid_channels, self.out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class SeqConv(nn.Module):
    def __init__(self, seq_type, in_planes, out_planes):
        super(SeqConv, self).__init__()

        self.type = seq_type
        self.in_channels = in_planes
        self.out_channels = out_planes

        if self.type == 'conv1x1-sobelx':
            self.conv0 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)

            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)

            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))

            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)

            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            self.conv0 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)

            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)

            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))

            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)

            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            self.conv0 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)

            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)

            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))

            # init mask
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = self.conv0(x)
        device = y0.get_device()
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        # pad with zero_vector
        b0_pad = torch.zeros(self.out_channels).view(1, -1, 1, 1).to(device)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias,
                      stride=1, groups=self.out_channels)
        return y1

    def rep_params(self):
        device = self.scale.get_device()
        if device < 0:
            device = None

        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=device)
        for i in range(self.out_channels):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        k0, b0 = self.conv0.switch_to_deploy()
        RK = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        RB = torch.ones(1, self.out_channels, 3, 3, device=device) * b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB


class OREPA_ECB(nn.Module):
    def __init__(self, in_planes, out_planes, depth_multiplier, act_type='prelu'):
        super(OREPA_ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.kernel_size = 3
        self.mid_channels = int(self.out_planes * depth_multiplier)

        with_idt = False

        if with_idt and (self.in_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        #   这里注意检测 orepa 当中的卷积是否进行了padding
        self.conv3x3 = conv_bn(self.in_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = OREPA_1x1_3x3('conv1x1_3x3', self.in_planes, self.out_planes, self.kernel_size, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv('conv1x1-sobelx', self.in_planes, self.out_planes,)
        self.conv1x1_sby = SeqConv('conv1x1-sobely', self.in_planes, self.out_planes,)
        self.conv1x1_lpl = SeqConv('conv1x1-laplacian', self.in_planes, self.out_planes,)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)

        else:
            RK, RB = self.rep_params()
            # 这里注意检查 OREPA 当中的forward 是如何写的，在训练阶段使用的是 ori_1x1_reparam()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):

        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()

        device = K3.get_device()
        if device < 0:
            device = None
        K0, B0 = self.conv3x3.switch_to_deploy()
        # K1_1, B1_1 = self.conv1x1_3x3.conv1.or1x1_reparam.weight, self.conv1x1_3x3.conv1.or1x1_reparam.bias
        # K1_3, B1_3 = self.conv1x1_3x3.conv2.orepa_reparam.weight, self.conv1x1_3x3.conv2.orepa_reparam.bias
        # reparam conv1x1_3x3
        k1_1, b1_1 = self.conv1x1_3x3.conv1.switch_to_deploy()
        k1_3, b1_3 = self.conv1x1_3x3.conv2.switch_to_deploy()
        K1 = F.conv2d(input=k1_3, weight=k1_1.permute(1, 0, 2, 3))
        B1 = torch.ones(1, self.mid_channels, 3, 3, device=device) * b1_1.view(1, -1, 1, 1)
        B1 = F.conv2d(input=B1, weight=k1_3).view(-1,) + b1_3


        RK, RB = (K0 + K1+ K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB
