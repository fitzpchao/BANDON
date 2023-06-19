import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses import accuracy
from mmseg.ops import resize
@HEADS.register_module()
class ForwardHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,

                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        super(ForwardHead, self).__init__( in_channels=1,
                 channels=1,
                 num_classes=1,**kwargs)


    def forward(self, inputs):
        """Forward function."""
        output = self._transform_inputs(inputs)

        return output
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight_map=None):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None and seg_weight_map is None:
            seg_weight = self.sampler.sample(seg_logit, seg_label).float()
        elif self.sampler is None and seg_weight_map is not None:
            seg_weight = seg_weight_map.float()
            seg_weight = seg_weight.squeeze(1)
        elif self.sampler is not None and seg_weight_map is not None:
            seg_weight_sample = self.sampler.sample(seg_logit, seg_label).float()
            seg_weight = seg_weight_sample * seg_weight_map.squeeze(1).float()
        else:
            seg_weight = None

        if self.is_reg:
            loss['loss_reg'] = self.loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight)
#           loss['acc_seg'] = accuracy(seg_logit, seg_label)
        else:
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

            if self.sampler is not None and seg_weight_map is None:
                seg_weight = self.sampler.sample(seg_logit, seg_label).float()
            elif self.sampler is None and seg_weight_map is not None:
                seg_weight = seg_weight_map.float()
                seg_weight = seg_weight.squeeze(1)
            elif self.sampler is not None and seg_weight_map is not None:
                seg_weight_sample = self.sampler.sample(seg_logit, seg_label).float()
                seg_weight = seg_weight_sample * seg_weight_map.squeeze(1).float()
            else:
                seg_weight = None

            seg_label = seg_label.squeeze(1)
            loss['loss_seg'] = self.loss_decode(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
            loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss