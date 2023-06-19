import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class CLSHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        # The simpliest classifier head
        super(CLSHead, self).__init__(**kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.cls_seg(x)
        return output
        