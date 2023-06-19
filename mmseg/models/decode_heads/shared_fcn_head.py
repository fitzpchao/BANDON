import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from ..builder import build_loss
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SharedFCNHead(BaseDecodeHead):
    """
    Fully Convolution Networks for Semantic Segmentation.
    """
    #TODO: Description of SharedFCNHead.

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 src_classes=None,
                 loss_decodes=None,
                 **kwargs):
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        assert isinstance(src_classes, list)
        self.src_classes = src_classes
        if isinstance(loss_decodes, list):
            assert len(loss_decodes) == len(src_classes)
            self.loss_decode = None
            self.loss_decodes = [build_loss(loss_decode) for loss_decode in loss_decodes]
        else:
            assert len(set(src_classes)) == 1 # classes of all source dataset are the same
            self.loss_decode = loss_decodes
            self.loss_decodes = None
        super(SharedFCNHead, self).__init__(loss_decode=self.loss_decode, **kwargs)
        assert self.num_classes == sum(self.src_classes) - len(self.src_classes) + 1

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,
                      src_ind, **kargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            src_ind (int): Index of source dataset for current input.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)  # C: self.num_classes
        assert seg_logits.shape[1] == self.num_classes

        non_bg_classes = [n-1 for n in self.src_classes]

        fg_bgn_ind = 1 + sum(non_bg_classes[:src_ind])
        fg_end_ind = 1 + sum(non_bg_classes[:src_ind+1])
        fg = seg_logits[:, fg_bgn_ind:fg_end_ind, :, :]
        bg = torch.cat((seg_logits[:, :fg_bgn_ind, :, :],
                        seg_logits[:, fg_end_ind:, :, :]), 1)
        bg, _ = torch.max(bg, dim=1, keepdim=True)
        seg_logits = torch.cat((bg, fg), 1)
        assert seg_logits.shape[0] == gt_semantic_seg.shape[0]

        if self.loss_decodes is not None:
            self.loss_decode = self.loss_decodes[src_ind]
        if 'seg_weight_map' in kargs.keys():
            losses = self.losses(seg_logits, gt_semantic_seg,
                                 seg_weight_map=kargs['seg_weight_map'])
        else:
            losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
