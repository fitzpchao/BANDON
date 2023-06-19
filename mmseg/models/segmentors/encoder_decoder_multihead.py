import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from mmseg.datasets.pipelines import OrientationUtil

@SEGMENTORS.register_module()
class EncoderDecoderMultiHead(EncoderDecoder):
    """Encoder Decoder segmentors with multiple heads.

    EncoderDecoder typically consists of backbone, several decode_heads (List),
    auxiliary_head.

    Args:
        decode_head (list): List of decode_heads.
        gt_keys (list[str]): Keys of gt maps for the corresponding decode heads,
            such as 'gt_semantic_seg', 'gt_edge_seg'.
        PALETTES (list[list[int] | None] | None): The palettes of the
            segmentation map for the corresponding decode heads. If None is given,
            no colorful map will be saved for the decode head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 gt_keys,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 PALETTES=None):
        super(EncoderDecoderMultiHead, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        assert self.num_heads == len(gt_keys)
        self.gt_keys = gt_keys

        if PALETTES is not None:
            assert isinstance(PALETTES, list)
            assert len(PALETTES) == self.num_heads
            self.PALETTES = PALETTES
            for i in range(self.num_heads):
                if PALETTES[i] is not None:
                    assert len(PALETTES[i]) == decode_head[i].num_classes
        self.PALETTES = PALETTES
        assert self.with_decode_head

        self.postprocess_keys = self.test_cfg.get('postprocess_keys', None)
        if self.postprocess_keys is not None:
            assert isinstance(self.postprocess_keys, list)
            assert len(self.postprocess_keys) == self.num_heads
            for i in range(self.num_heads):
                gt_key = self.gt_keys[i]
                pp_keys = self.postprocess_keys[i]
                assert isinstance(pp_keys, list)
                for j, pp_key in enumerate(pp_keys):
                    if     (gt_key.startswith('gt_orient_edge_')
                        and gt_key.endswith('_seg')
                        and pp_key.startswith('gt_orient_edge_')
                        and pp_key.endswith('_seg')
                    ):
                        try:
                            num_bin = int(gt_keys[i].split('_')[3])
                        except:
                            raise ValueError(f"A NUMBER is expected at " \
                                + f"gt_orient_edge_*_seg' for gt_keys[{i}]")
                        self.ort_util = OrientationUtil(num_bin)
                        try:
                            pp_num_bin = int(pp_key.split('_')[3])
                        except:
                            raise ValueError(f"A NUMBER is expected at " \
                                + f"'gt_orient_edge_*_seg' for postprocess_keys[{i}][{j}]")
                        self.pp_ort_util = OrientationUtil(pp_num_bin)
                    else:
                        raise NotImplementedError

    def get_palettes(self):
        return self.PALETTES

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        self.num_heads = len(decode_head)
        self.decode_head = nn.ModuleList()
        for i in range(self.num_heads):
            self.decode_head.append(builder.build_head(decode_head[i]))

        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = []
        for i  in range(self.num_heads):
            # align_corners options are set to be the same for all decode_heads
            assert self.align_corners == self.decode_head[i].align_corners
            self.num_classes.append(self.decode_head[i].num_classes)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_heads):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()


    def _decode_head_forward_train(self, x, img_metas, gt_seg, head_ind, seg_weight_map=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss = self.decode_head[head_ind].forward_train(x, img_metas, gt_seg, self.train_cfg,
                                                       seg_weight_map=seg_weight_map)

        losses.update(add_prefix(loss, f'decode_{head_ind}'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        predictions = []
        for decode_head in self.decode_head:
            predictions.append(decode_head.forward_test(x, img_metas, self.test_cfg))

        return torch.cat(predictions, 1)


    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert(self.num_heads == len(kwargs),
               f'num_heads = {self.num_heads} \n',
               f'len(kwargs) = {len(kwargs)}')

        for gt_key in self.gt_keys:
            assert gt_key in kwargs

        x = self.extract_feat(img)

        losses = dict()

        for i in range(self.num_heads):
            gt_key = self.gt_keys[i]

            if 'weightmaps' in kwargs:
                seg_weight_map = kwargs['weightmaps'].get(gt_key, None)
            else:
                seg_weight_map = None

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          kwargs[gt_key], head_ind=i,
                                                          seg_weight_map=seg_weight_map)
            losses.update(loss_decode)

        if self.with_auxiliary_head:
            if 'weightmaps' in kwargs:
                seg_weight_map = kwargs['weightmaps'].get('gt_semantic_seg', None)
            else:
                seg_weight_map = None
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight_map=seg_weight_map)
            losses.update(loss_aux)

        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The list of output prob map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_logit = seg_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_logit = seg_logit.flip(dims=(2, ))

        indexes = [sum(self.num_classes[:y]) for y in range(self.num_heads+1)]
        outputs = []
        for i in range(self.num_heads):
            if self.decode_head[i].is_reg:
                outputs.append(seg_logit[:, indexes[i]:indexes[i+1], ...])
            else:
                outputs.append(
                    F.softmax(seg_logit[:, indexes[i]:indexes[i+1], ...], dim=1))
        return outputs

    def _permute_list_of_lists(self, src):
        batchsize = len(src[0])
        out = [[] for _ in range(batchsize)]
        for i in range(len(src)):
            for b in range(batchsize):
                out[b].append(src[i][b])
        return out
    def _postprocess(self, seg_pred, i):
        if self.postprocess_keys is None:
            return []
        gt_key = self.gt_keys[i]
        pp_keys = self.postprocess_keys[i]
        assert isinstance(pp_keys, list)
        pp_outs = []
        for j, pp_key in enumerate(pp_keys):
            if     (gt_key.startswith('gt_orient_edge_')
                and gt_key.endswith('_seg')
                and pp_key.startswith('gt_orient_edge_')
                and pp_key.endswith('_seg')
            ):
                interpolated_orient_angle = \
                    self.ort_util.get_interpolated_angle(seg_pred.cpu().numpy())
                interpolated_edge_orient_label = \
                    self.pp_ort_util.angle_to_label(interpolated_orient_angle)

                seg_pred = seg_pred.argmax(dim=1).cpu().numpy()
                interpolated_edge_orient_label[seg_pred == 0] = 0

                pp_outs.append(interpolated_edge_orient_label)
            else:
                raise NotImplementedError
        return pp_outs

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        predictions = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            return predictions
        seg_preds = []
        reg_preds = []
        pp_preds = []
        seg_preds.append(predictions[0].cpu().detach().numpy())
        for i in range(1, self.num_heads):
            pp_pred = self._postprocess(predictions[i], i)
            if pp_pred:
                pp_preds.append(*pp_pred)
            if self.decode_head[i].is_reg:
                reg_preds.append(list(predictions[i].cpu().detach().numpy()))
            else:
                seg_pred = predictions[i].argmax(dim=1)
                seg_pred = seg_pred.cpu().detach().numpy()
                # unravel batch dim
                seg_pred = list(seg_pred)
                seg_preds.append(seg_pred)
        return self._permute_list_of_lists(seg_preds + reg_preds + pp_preds)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented prediction inplace
        predictions = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_predictions = self.inference(imgs[i], img_metas[i], rescale)
            for j in range(self.num_heads):
                predictions[j] += cur_predictions[j]
        seg_preds = []
        pp_preds = []
        seg_preds.append((predictions[0]/len(imgs)).cpu().detach().numpy())
        for j in range(1, self.num_heads):
            predictions[j] /= len(imgs)
            pp_pred = self._postprocess(predictions[j], j)
            if pp_pred:
                pp_preds.append(*pp_pred)
            if self.decode_head[j].is_reg:
                raise NotImplementedError('Regresssion is not supported for aug_test')
            else:
                seg_pred = predictions[j].argmax(dim=1)
                seg_pred = seg_pred.cpu().detach().numpy()
                # unravel batch dim
                seg_pred = list(seg_pred)
                seg_preds.append(seg_pred)
        return self._permute_list_of_lists(seg_preds + pp_preds)
