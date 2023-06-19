import torch.nn as nn
from mmcv.cnn import (constant_init, build_norm_layer,
                      kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch
import torch.nn.functional as F
from ..builder import BACKBONES
from .resnet import ResNet, ResNetV1c
from mmcv.cnn import ConvModule
from torch.nn.modules.padding import ReplicationPad2d

class PPM(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        in_dim (int): Input channels.
        reduction_dim (int): Channels after modules, before conv_seg.
        bins (tuple[int]): List of bin numbers in PPM.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        drop_out (float, optional): Dropout ratio. Default: 0.2.
        return_patch (bool): Whether to return the patch feature vectors.
            Default: True.
    """

    def __init__(self,
                 in_dim,
                 reduction_dim,
                 bins,
                 conv_cfg,
                 norm_cfg,
                 dropout=0.2):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    ConvModule(
                        in_dim,
                        reduction_dim,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        patch_f = []
        for f in self.features:
            temp = f(x)
            out.append(
                F.interpolate(
                    temp, x_size[2:], mode='bilinear', align_corners=True))
            patch_f.append(temp)
        out = torch.cat(out, 1)
        return out


@BACKBONES.register_module()
class SiamUnet_conc(nn.Module):

    def __init__(self,
                 use_IN1=False,
                 input_nbr=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 **kwargs):
        super(SiamUnet_conc, self).__init__()

        self.use_IN1 = use_IN1



        self.norm_eval = norm_eval


        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        _ , self.bn11 = build_norm_layer(norm_cfg,16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.bn12 = nn.BatchNorm2d(16)
        _, self.bn12 = build_norm_layer(norm_cfg, 16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn21 = nn.BatchNorm2d(32)
        _, self.bn21 = build_norm_layer(norm_cfg, 32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.bn22 = nn.BatchNorm2d(32)
        _, self.bn22 = build_norm_layer(norm_cfg, 32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn31 = nn.BatchNorm2d(64)
        _, self.bn31 = build_norm_layer(norm_cfg, 64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn32 = nn.BatchNorm2d(64)
        _, self.bn32 = build_norm_layer(norm_cfg, 64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn33 = nn.BatchNorm2d(64)
        _, self.bn33 = build_norm_layer(norm_cfg, 64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn41 = nn.BatchNorm2d(128)
        _, self.bn41 = build_norm_layer(norm_cfg, 128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.bn42 = nn.BatchNorm2d(128)
        _, self.bn42 = build_norm_layer(norm_cfg, 128)

        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.bn43 = nn.BatchNorm2d(128)
        _, self.bn43 = build_norm_layer(norm_cfg, 128)

        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(384, 128, kernel_size=3, padding=1)
        # self.bn43d = nn.BatchNorm2d(128)
        _, self.bn43d = build_norm_layer(norm_cfg, 128)

        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        # self.bn42d = nn.BatchNorm2d(128)
        _, self.bn42d = build_norm_layer(norm_cfg, 128)

        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        # self.bn41d = nn.BatchNorm2d(64)
        _, self.bn41d = build_norm_layer(norm_cfg, 64)

        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)
        # self.bn33d = nn.BatchNorm2d(64)
        _, self.bn33d = build_norm_layer(norm_cfg, 64)

        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        # self.bn32d = nn.BatchNorm2d(64)
        _, self.bn32d = build_norm_layer(norm_cfg, 64)

        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        # self.bn31d = nn.BatchNorm2d(32)
        _, self.bn31d = build_norm_layer(norm_cfg, 32)

        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(96, 32, kernel_size=3, padding=1)
        # self.bn22d = nn.BatchNorm2d(32)
        _, self.bn22d = build_norm_layer(norm_cfg, 32)

        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        # self.bn21d = nn.BatchNorm2d(16)
        _, self.bn21d = build_norm_layer(norm_cfg, 16)

        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(48, 16, kernel_size=3, padding=1)
        # self.bn12d = nn.BatchNorm2d(16)
        _, self.bn12d = build_norm_layer(norm_cfg, 16)

        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)


        if self.use_IN1:
            self.bn11=nn.InstanceNorm2d(16, affine=True)


    def init_weights(self, pretrained=None):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m,
                            (_BatchNorm, nn.GroupNorm, nn.InstanceNorm2d)):
                constant_init(m, 1)
        if self.use_IN1:
            for m in self.layer0:
                if isinstance(m, nn.InstanceNorm2d):
                    constant_init(m, 1)





    def forward(self, x1, x2):
        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        ####################################################
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)


        return [x11d]

    def train(self, mode=True):
        super(SiamUnet_conc, self).train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
