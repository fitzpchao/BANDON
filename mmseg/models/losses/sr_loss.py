import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

from ..builder import LOSSES
from .utils import weighted_loss
# from ...datasets.vis import show

from torch.fft import irfft  as irfft
from torch.fft import fft as rfft

def normkernel_to_downkernel(rescaled_blur_hr, rescaled_hr, ksize, eps=1e-10):
    # blur_img = rfft(rescaled_blur_hr, 3, onesided=False)
    blur_img = torch.view_as_real(torch.fft.fftn(rescaled_blur_hr, dim=(1,2,3)))
    # img = rfft(rescaled_hr, 3, onesided=False)
    img = torch.view_as_real(torch.fft.fftn(rescaled_hr, dim=(1,2,3)))

    denominator = img[:, :, :, :, 0] * img[:, :, :, :, 0] + img[:, :, :, :, 1] * img[:, :, :, :, 1] + eps
    # denominator[denominator==0] = eps

    inv_denominator = torch.zeros_like(img)
    inv_denominator[:, :, :, :, 0] = img[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -img[:, :, :, :, 1] / denominator

    kernel = torch.zeros_like(blur_img).cuda()
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 0] \
                            - inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 1] \
                            + inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 0]

    ker = convert_otf2psf(kernel, ksize)

    return ker
def convert_otf2psf(otf, size):
    ker = torch.zeros(size).cuda()
    psf = torch.fft.irfftn(torch.view_as_complex(otf), s=(otf.shape[1],otf.shape[2],otf.shape[3]), dim=(1,2,3))

    # circularly shift
    ksize = size[-1]
    centre = ksize//2 + 1


    ker[:, :, (centre-1):, (centre-1):] = psf[:, :, :centre, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, (centre-1):, :(centre-1)] = psf[:, :, :centre, -(centre-1):]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), (centre-1):] = psf[:, :, -(centre-1):, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]#.mean(dim=1, keepdim=True)

    return ker
def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    pc = k.shape[-1]//2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    # Clip values lower than the minimum value
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    # Normalize to sum to 1
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k

@weighted_loss
def correctionloss(pred,
             target,eps=1e-6):
    """Warpper of correction loss."""
    lr_blured,lr = target[:,:3],target[:,3:]
    # raise Exception( pred.size(),target.size())

    ks = []
    mask = torch.ones_like(pred).cuda()
    for c in range(lr_blured.shape[1]):
        k_correct = normkernel_to_downkernel(lr_blured[:, c:c + 1, ...], lr[:, c:c + 1, ...], pred.size(), eps)
        ks.append(k_correct.clone())
        mask *= k_correct
    ks = torch.cat(ks, dim=1)
    k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask > 0)
    k_correct = zeroize_negligible_val(k_correct, n=40)
    loss = nn.functional.l1_loss(pred,k_correct)
    return loss


@LOSSES.register_module()
class CORRLoss(nn.Module):
    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.eps=eps
        self.reduction=reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                **kwargs):
        loss = self.loss_weight * correctionloss(
            pred,
            target,
            eps=self.eps,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor)

        return loss
