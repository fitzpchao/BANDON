import torch.nn as nn
from mmcv.cnn import (constant_init,build_norm_layer,build_conv_layer,
                      kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch
import torch.nn.functional as F
from ..builder import BACKBONES
from .resnet import ResNet, ResNetV1c, Bottleneck
from mmcv.cnn import ConvModule


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

class TaskLayer(nn.Module):
    def __init__(self, input_channels, out_channels,norm_cfg,conv_cfg=None):
        super(TaskLayer, self).__init__()
        self.layers= nn.Sequential(
            ConvModule(
                        input_channels,
                        out_channels,
                        3,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))
        )
    def forward(self, x):
        x = self.layers(x)
        return x


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels,out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)

@BACKBONES.register_module()
class ChangePSPNetMTL(nn.Module):
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
                 ppm_bins=(1, 2, 3, 6),
                 pretrained=None,
                 stem_channels=64,
                 base_channels=64,
                 deep_stem=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 **kwargs):
        super(ChangePSPNetMTL, self).__init__()
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



        self.conv_down = nn.Conv2d(
            4096, 512, kernel_size=3, padding=1, bias=True)

        self.ppm = PPM(
            2048,
            int(2048 / len(ppm_bins)),
            ppm_bins,
            conv_cfg,
            norm_cfg)
        self.norm_eval = norm_eval

        self.cls_offsetbuild = TaskLayer(512, 512,  norm_cfg=norm_cfg)
        self.cls_seg = TaskLayer(512, 512,  norm_cfg=norm_cfg)

        self.cls_cd = TaskLayer(1024,512,norm_cfg=norm_cfg)
        self.cls_offsetcd = TaskLayer(1024, 512,  norm_cfg=norm_cfg)

        self.sablock_offsetbuild = SABlock(1024, 256)
        self.sablock_offsetcd = SABlock(512, 256)
        self.sablock_seg = SABlock(1024,256)

        self.out_project = nn.Sequential(ConvModule(256 * 3,
                                                512,
                                                1,
                                                conv_cfg=conv_cfg,
                                                norm_cfg=norm_cfg,
                                                act_cfg=dict(type='ReLU')),
                                     )


    def init_weights(self, pretrained=None):

        need_init_modules = [
            self.conv_down.modules(),
            self.ppm.modules(),
            self.cls_seg.modules(),
            self.cls_offsetbuild.modules(),
            self.cls_offsetcd.modules(),
            self.cls_cd.modules(),

            self.sablock_seg.modules(),
            self.sablock_offsetbuild.modules(),
            self.sablock_offsetcd.modules(),

            self.out_project.modules()
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        return x


    def forward(self, x1, x2):

        x1 = self._forward_embedding(x1)
        x2 = self._forward_embedding(x2)

        x1 = self.conv_down(x1)
        x2 = self.conv_down(x2)



        x_cat = torch.cat([x1, x2], 1)

        f_flowst1 = self.cls_offsetbuild(x1)
        f_flowst2 = self.cls_offsetbuild(x2)
        f_seg1 = self.cls_seg(x1)
        f_seg2 = self.cls_seg(x2)
        f_flowbt = self.cls_offsetcd(x_cat)
        f_cd = self.cls_cd(x_cat)

        att_seg = self.sablock_seg(torch.cat([f_seg1,f_seg2],1))
        att_flowst = self.sablock_offsetbuild(torch.cat([f_flowst1,f_flowst2],1))
        att_flowbt = self.sablock_offsetcd(f_flowbt)

        att_feat = self.out_project(torch.cat([att_seg,att_flowst,att_flowbt],1))


        f_final = torch.cat([f_cd,att_feat],1)


        return [f_final,f_seg1,f_seg2,f_flowbt,f_flowst1,f_flowst2]

    def train(self, mode=True):
        super(ChangePSPNetMTL, self).train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
