import torch.nn as nn
import torch.nn.functional as F
import torch
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class CDNet(BaseSegmentor):
    """Change segmentor (two aligned image per sample as input and one label 
       mask as label) with extra task (one image per sample as input).

    ChangeExtraSegmentor typically consists of backbone, decode_head, extra_head, 
    shared_extra_head.
    Note that shared_extra_head and extra_head is only used for deep supervision during training,
    which could be dumped during inference.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        nips (int, optional): number of images per sample for main task (change
            segmentation). Default: 2
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 gt_keys=['gt_semantic_seg'],
                 nips=2,
                 neck=None,
                 # extra_head=None,
                 # shared_extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CDNet, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        self.nips = nips

        assert isinstance(gt_keys, list)
        assert self.num_heads == len(gt_keys)
        self.gt_keys = gt_keys

        assert self.with_decode_head

        if nips != 2:
            raise NotImplementedError

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

    def _init_extra_head(self, extra_head):
        """Initialize ``extra_head``"""
        if extra_head is None:
            self.extra_head = None
            self.num_extra_heads = 0
            return
        assert isinstance(extra_head, list)
        self.num_extra_heads = len(extra_head)
        self.extra_head = nn.ModuleList()
        for i in range(self.num_extra_heads):
            self.extra_head.append(builder.build_head(extra_head[i]))

    def _init_shared_extra_head(self, shared_extra_head):
        """Initialize ``shared_extra_head``"""
        if shared_extra_head is None:
            self.shared_extra_head = None
        else:
            self.shared_extra_head = builder.build_head(shared_extra_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(CDNet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_heads):
            self.decode_head[i].init_weights()



    def extract_feat(self, img):
        """Extract features from images."""
        assert self.nips == 2
        assert img.size(1) == 3 * self.nips

        x = self.backbone(img[:, 0:3, :, :], img[:, 3:6, :, :])
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        # extra_data = None
        # if self.output_shared_extra_head:
        #     extra_data = torch.cat((img[:, 0:3, :, :], img[:, 3:6, :, :]), dim=0)
        #     N = img.size(0)

        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # # ---- pair flip and average
        # img_pf = torch.cat((img[:, 3:6, :, :], img[:, 0:3, :, :]), dim=1)
        # x_pf, _ = self.extract_feat(img_pf)
        # out_pf = self._decode_head_forward_test(x_pf, img_metas)
        # out_pf = resize(
        #     input=out_pf,
        #     size=img_pf.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # out = 0.5 * out + 0.5 * out_pf

        return out
    def slide_inference_mtl(self, img, img_metas,indexs_list=[1],resize_list=[True],num_classes_list=[2,3],scale=4.0):
        """Inference by sliding-window with overlap."""
        img =img[0].cuda()
        img_meta =img_metas[0]
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')

        # num_classes = self.num_classes[0]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        len_out=len(indexs_list)
        pred_list = []
        for i in range(len_out):
            preds = img.new_zeros((batch_size, num_classes_list[i], h_img, w_img))
            pred_list.append(preds)
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        key = (h_crop, w_crop, h_stride, w_stride)
        if key in self.crop_weight_dict:
            crop_weight = self.crop_weight_dict[key]
        else:
            crop_weight = img.new_ones((h_crop, w_crop), dtype=torch.float32)
            h_thres = h_crop - h_stride
            w_thres = w_crop - w_stride
            for i in range(h_crop):
                weight_i = 1.0
                if i < h_thres:
                    weight_i *= (1.0 * (i +1)/h_thres)
                if i > h_stride-1:
                    weight_i *= (1.0 * (h_crop - i )/h_thres)
                for j in range(w_crop):
                    weight_j = 1.0 * weight_i
                    if j < w_thres:
                        weight_j *= (1.0 * (j +1 )/w_thres)
                    if j > w_stride-1:
                        weight_j *= (1.0 * (w_crop - j )/w_thres)
                    crop_weight[i][j] *= weight_j
            self.crop_weight_dict[key] = crop_weight


        crop_weight_mat_list = []
        for i in range(len_out):
            crop_weight_mat = crop_weight.repeat((batch_size, num_classes_list[i], 1, 1))
            crop_weight_mat_list.append(crop_weight_mat)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                y2_2 = int(min(y1 // scale + h_crop // scale, h_img // scale))
                x2_2 = int(min(x1 // scale + w_crop // scale , w_img // scale))
                y1_2 = int(max(y2 // scale - h_crop // scale, 0))
                x1_2 = int(max(x2 // scale - w_crop // scale, 0))

                crop_img = img[:, :3, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                # raise Exception(y1_2,y2_2,x1_2,x2_2)
                crop_img2 = img[:, 3:, y1_2:y2_2, x1_2:x2_2]
                pad_img2 = crop_img2.new_zeros(
                    (crop_img2.size(0), crop_img2.size(1), h_crop//4, w_crop//4))
                pad_img2[:, :, :y2_2 - y1_2, :x2_2 - x1_2] = crop_img2

                pad_img2_up = torch.zeros(pad_img.size(),dtype=torch.float32).cuda()
                pad_img2_up[:,:,:h_crop//4,:w_crop//4]= pad_img2

                pad_img_cat = torch.cat([pad_img,pad_img2_up],1)


                pad_logit_list = self.encode_decode_mtl(pad_img_cat, img_meta,indexs_list=indexs_list,resize_list=resize_list)
                for i in range(len_out):
                    # if 1:
                    #     raise Exception(pred_list[i].size(),pad_logit_list[i].size(),crop_weight_mat_list[i].size())
                    pred_list[i][:, :, y1:y2,
                    x1:x2] += pad_logit_list[i][:, :, :y2 - y1, :x2 - x1] * crop_weight_mat_list[i]
                count_mat[:, :, y1:y2, x1:x2] += crop_weight
        assert (count_mat == 0).sum() == 0
        for i in range(len_out):
            pred_list[i] = pred_list[i] / count_mat

        return pred_list
    def slide_inference_restoration(self, img, img_metas,indexs_list=[1],resize_list=[True],num_classes_list=[3]):
        """Inference by sliding-window with overlap."""
        img =img[0].cuda()
        img_meta =img_metas[0]
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')

        # num_classes = self.num_classes[0]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        len_out=len(indexs_list)
        pred_list = []
        for i in range(len_out):
            preds = img.new_zeros((batch_size, num_classes_list[i], h_img, w_img))
            pred_list.append(preds)
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        key = (h_crop, w_crop, h_stride, w_stride)
        if key in self.crop_weight_dict:
            crop_weight = self.crop_weight_dict[key]
        else:
            crop_weight = img.new_ones((h_crop, w_crop), dtype=torch.float32)
            h_thres = h_crop - h_stride
            w_thres = w_crop - w_stride
            for i in range(h_crop):
                weight_i = 1.0
                if i < h_thres:
                    weight_i *= (1.0 * (i +1)/h_thres)
                if i > h_stride-1:
                    weight_i *= (1.0 * (h_crop - i )/h_thres)
                for j in range(w_crop):
                    weight_j = 1.0 * weight_i
                    if j < w_thres:
                        weight_j *= (1.0 * (j +1 )/w_thres)
                    if j > w_stride-1:
                        weight_j *= (1.0 * (w_crop - j )/w_thres)
                    crop_weight[i][j] *= weight_j
            self.crop_weight_dict[key] = crop_weight


        crop_weight_mat_list = []
        for i in range(len_out):
            crop_weight_mat = crop_weight.repeat((batch_size, num_classes_list[i], 1, 1))
            crop_weight_mat_list.append(crop_weight_mat)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # y2_2 = int(min(y1  + h_crop , h_img ))
                # x2_2 = int(min(x1  + w_crop , w_img ))
                # y1_2 = int(max(y2  - h_crop , 0))
                # x1_2 = int(max(x2  - w_crop , 0))

                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_logit_list = self.encode_decode_mtl(pad_img, img_meta,indexs_list=indexs_list,resize_list=resize_list)
                for i in range(len_out):

                    pred_list[i][:, :, y1:y2,
                    x1:x2] += pad_logit_list[i][:, :, :y2 - y1, :x2 - x1] * crop_weight_mat_list[i]
                count_mat[:, :, y1:y2, x1:x2] += crop_weight
        assert (count_mat == 0).sum() == 0
        for i in range(len_out):
            pred_list[i] = pred_list[i] / count_mat
        return pred_list

    def encode_decode_mtl(self, img, img_metas,indexs_list=[1],resize_list=[True]):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        # extra_data = None
        # if self.output_shared_extra_head:
        #     extra_data = torch.cat((img[:, 0:3, :, :], img[:, 3:6, :, :]), dim=0)
        #     N = img.size(0)
        out_list=[]
        x = self.extract_feat(img)
        # if 1:
        #     raise Exception(indexs_list)
        for i in range(len(indexs_list)):
            out = self.decode_head[indexs_list[i]].forward_test(x, img_metas, self.test_cfg)
            if resize_list[i]:
                out = resize(
                    input=out,
                    size=img.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            out_list.append(out)


        return out_list

    def _decode_head_forward_train(self, x, img_metas, gt_seg, head_ind, seg_weight_map=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss = self.decode_head[head_ind].forward_train(x, img_metas, gt_seg,
                                                       self.train_cfg,
                                                       seg_weight_map=seg_weight_map)

        losses.update(add_prefix(loss, f'decode_{head_ind}'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _shared_extra_head_forward_train(self, x, img_metas, gt_seg, src_ind, seg_weight_map=None):
        """Run forward function and calculate loss for the shared extra head in
        training.
        """
        losses = dict()
        loss = self.shared_extra_head.forward_train(x, img_metas, gt_seg,
                                                    self.train_cfg, src_ind,
                                                    seg_weight_map=seg_weight_map)

        losses.update(add_prefix(loss, f'shared_extra_{src_ind}'))
        return losses

    def _shared_extra_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.shared_extra_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def forward_dummy(self, img, extra_data=None):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            extra_data (dict, optional): Image and gt for the extra task.
                Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

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
                seg_weight_map=seg_weight_map
            )
            losses.update(loss_decode)



        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')

        num_classes = self.num_classes[0]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        key = (h_crop, w_crop, h_stride, w_stride)
        if key in self.crop_weight_dict:
            crop_weight = self.crop_weight_dict[key]
        else:
            crop_weight = img.new_ones((h_crop, w_crop), dtype=torch.float32)
            h_thres = h_crop - h_stride
            w_thres = w_crop - w_stride
            for i in range(h_crop):
                weight_i = 1.0
                if i < h_thres:
                    weight_i *= (1.0 * (i +1)/h_thres)
                if i > h_stride-1:
                    weight_i *= (1.0 * (h_crop - i )/h_thres)
                for j in range(w_crop):
                    weight_j = 1.0 * weight_i
                    if j < w_thres:
                        weight_j *= (1.0 * (j +1 )/w_thres)
                    if j > w_stride-1:
                        weight_j *= (1.0 * (w_crop - j )/w_thres)
                    crop_weight[i][j] *= weight_j
            self.crop_weight_dict[key] = crop_weight

        crop_weight_mat = crop_weight.repeat((batch_size, num_classes, 1, 1))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.encode_decode(pad_img, img_meta)
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1] * crop_weight_mat
                count_mat[:, :, y1:y2, x1:x2] += crop_weight
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

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
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_logit = seg_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_logit = seg_logit.flip(dims=(2, ))

        # if self.output_shared_extra_head:
        #     c0 = self.num_classes[0]
        #     c1 = self.num_classes[0] + self.shared_extra_head.num_classes
        #     outputs = [
        #         F.softmax(seg_logit[:,  :c0,...], dim=1),
        #         F.softmax(seg_logit[:,c0:c1,...], dim=1),
        #         F.softmax(seg_logit[:,c1:  ,...], dim=1),
        #     ]
        # else:
        outputs = [F.softmax(seg_logit, dim=1)]

        return outputs

    def _permute_list_of_lists(self, src):
        batchsize = len(src[0])
        out = [[] for _ in range(batchsize)]
        for i in range(len(src)):
            for b in range(batchsize):
                out[b].append(src[i][b]) 
        return out

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        predictions = self.inference(img, img_meta, rescale)
        # if self.output_shared_extra_head:
        #     assert len(predictions) == 3
        # else:
        assert len(predictions) == 1
        if torch.onnx.is_in_onnx_export():
            return predictions[0]
        seg_preds = []
        seg_preds.append(predictions[0].cpu().detach().numpy())
        # if self.output_shared_extra_head:
        #     seg_preds.append(
        #         list(predictions[1].argmax(dim=1).cpu().detach().numpy()))
        #     seg_preds.append(
        #         list(predictions[2].argmax(dim=1).cpu().detach().numpy()))
        return self._permute_list_of_lists(seg_preds)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented predictions inplace
        predictions = self.inference(imgs[0], img_metas[0], rescale)
        # if self.output_shared_extra_head:
        #     assert len(predictions) == 3
        # else:
        assert len(predictions) == 1
        for i in range(1, len(imgs)):
            cur_predictions = self.inference(imgs[i], img_metas[i], rescale)
            for j in range(len(predictions)):
                predictions[j] += cur_predictions[j]
        seg_preds = []
        seg_preds.append((predictions[0]/len(imgs)).cpu().detach().numpy())
        # if self.output_shared_extra_head:
        #     seg_preds.append(
        #         list(predictions[1]/len(imgs).argmax(dim=1).cpu().detach().numpy()))
        #     seg_preds.append(
        #         list(predictions[2]/len(imgs).argmax(dim=1).cpu().detach().numpy()))
        return self._permute_list_of_lists(seg_preds)
