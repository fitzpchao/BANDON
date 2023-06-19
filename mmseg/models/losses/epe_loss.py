import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

from ..builder import LOSSES
from .utils import weighted_loss
# from ...datasets.vis import show

@weighted_loss
def epe_loss(pred,
             target):
    """Warpper of mse loss."""
    # target中nan的地方为1,非nan的地方为0
    ignore_mask=torch.isnan(target).type(target.type())
    revserse_ignore_mask=1-ignore_mask
    pred=pred*revserse_ignore_mask
    target[torch.where(ignore_mask)]=0
    loss=torch.norm(pred-target,p=2,dim=1)
    # show.add_img(imgs=[pred[0],target[0],loss[0]],img_flags=['reg_result','reg_result','loss'])

    return loss


@LOSSES.register_module()
class EPELoss(nn.Module):
    """EPELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * epe_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        return loss
