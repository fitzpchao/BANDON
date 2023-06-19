import torch.nn as nn
from mmcv.cnn import (constant_init,
                      kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch
import torch.nn.functional as F
from ..builder import BACKBONES
from .resnet import ResNet, ResNetV1c
from mmcv.cnn import ConvModule,build_norm_layer

backbone_channels = [64*4,128*4,256*4,512*4]
class decoder_block(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg=dict(type='BN', requires_grad=True)):
        super(decoder_block, self).__init__()
        self.norm_cfg = norm_cfg

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels,in_channels//2,kernel_size=1),
                                    build_norm_layer(self.norm_cfg, in_channels // 2)[1],
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels//2,out_channels,kernel_size=1),
                                    build_norm_layer(self.norm_cfg, out_channels)[1],
                                    nn.ReLU(inplace=True))

    def forward(self,input):
        input = self.conv_1(input)
        input = self.conv_2(input)
        return input

@BACKBONES.register_module()
class Siam_conc_r50(nn.Module):
    """ResNet backbone for change segmentation (two aligned image per sample as
       input and one label mask as label) with extra task (one image per sample
       as input).

    Args:
        use_IN1 (bool): Whether to use IN after first conv of the network.
        ppm_bins (tuple[int], optional): List of bin numbers in PPM.
            Default: (1, 2, 3, 6).
        pretrained (str, optional): Path of pretrain model. Default: None,
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default" 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        conv_cfg (dict): The conv layer config. If None, 'Conv2d'. Default: None
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    def __init__(self,
                 use_IN1=True,
                 pretrained=None,
                 stem_channels=64,
                 base_channels=64,
                 deep_stem=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 **kwargs):
        super(Siam_conc_r50, self).__init__()
        resnet = ResNet(stem_channels=stem_channels,
                        base_channels=base_channels,
                        deep_stem=deep_stem,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        norm_eval=norm_eval,
                        **kwargs)
        resnet.init_weights(pretrained=pretrained)

        self.use_IN1 = use_IN1
        if self.use_IN1:
            if deep_stem:
                resnet.stem[1] = nn.InstanceNorm2d(stem_channels//2,
                                                   affine=True)
                self.layer0 = nn.Sequential(resnet.stem, resnet.maxpool)
            else:
                self.layer0 = nn.Sequential(resnet.conv1,
                                            nn.InstanceNorm2d(base_channels,
                                                affine=True),
                                            resnet.relu, resnet.maxpool)
        else:
            if deep_stem:
                self.layer0 = nn.Sequential(resnet.stem, resnet.maxpool)
            else:
                self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                            resnet.maxpool)

        self.layer1 = getattr(resnet, resnet.res_layers[0])
        self.layer2 = getattr(resnet, resnet.res_layers[1])
        self.layer3 = getattr(resnet, resnet.res_layers[2])
        self.layer4 = getattr(resnet, resnet.res_layers[3])



        self.decoder_4 = decoder_block(in_channels=backbone_channels[3] * 2, out_channels=backbone_channels[2],
                                       norm_cfg=norm_cfg)
        self.decoder_3 = decoder_block(in_channels=backbone_channels[2] * 3, out_channels=backbone_channels[1],
                                       norm_cfg=norm_cfg)
        self.decoder_2 = decoder_block(in_channels=backbone_channels[1] * 3, out_channels=backbone_channels[0],
                                       norm_cfg=norm_cfg)
        self.decoder_1 = decoder_block(in_channels=backbone_channels[0] * 3, out_channels=backbone_channels[0],
                                       norm_cfg=norm_cfg)

        self.out = nn.Sequential(
            nn.Conv2d(backbone_channels[0], backbone_channels[0], kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, backbone_channels[0])[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
           )


        self.norm_eval = norm_eval

    def init_weights(self, pretrained=None):

        need_init_modules = [
            self.decoder_1.modules(),
            self.decoder_2.modules(),
            self.decoder_3.modules(),
            self.decoder_4.modules(),
        ]
        for modules in need_init_modules:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m,
                                (_BatchNorm, nn.GroupNorm, nn.InstanceNorm2d)):
                    constant_init(m, 1)
        if self.use_IN1:
            for m in self.layer0:
                if isinstance(m, nn.InstanceNorm2d):
                    constant_init(m, 1)

    def _forward_embedding(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1,x2,x3,x4]



    def forward(self, x1, x2):
        T1_features= self._forward_embedding(x1)
        T2_features = self._forward_embedding(x2)
        F4_cat = torch.cat([T1_features[3], T2_features[3]], dim=1)
        F4 = self.decoder_4(F4_cat)  # H/32
        F4 = F.interpolate(F4, size=T1_features[2].shape[2:], mode="bilinear", align_corners=True)  # H/16

        F3_cat = torch.cat([T1_features[2], T2_features[2], F4], dim=1)
        F3 = self.decoder_3(F3_cat)  # H/16
        F3 = F.interpolate(F3, size=T1_features[1].shape[2:], mode="bilinear", align_corners=True)  # H/8

        F2_cat = torch.cat([T1_features[1], T2_features[1], F3], dim=1)
        F2 = self.decoder_2(F2_cat)  # H/8
        F2 = F.interpolate(F2, size=T1_features[0].shape[2:], mode="bilinear", align_corners=True)  # H/4

        F1_cat = torch.cat([T1_features[0], T2_features[0], F2], dim=1)
        F1 = self.decoder_1(F1_cat)  # H/4
        F1 = F.interpolate(F1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        x = self.out(F1)

        return [x]

    def train(self, mode=True):
        super(Siam_conc_r50, self).train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
