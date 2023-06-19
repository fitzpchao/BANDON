from random import choice
import mmcv
import numpy as np
from numpy import random
import cv2
from ..pipelines import flow_utils
from ..pipelines import seg_utils
from ..builder import PIPELINES
from ..pipelines import utils_sr
import torch
@PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            if self.img_scale is not None: # mode 3: given a range of image ratio
                # mode 1: given a scale and a range of image ratio
                assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is not None: 
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
            else:
                if isinstance(results['img'], list):
                    h, w = results['img'][0].shape[:2]
                else:
                    h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((h, w),
                                                            self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]
        if self.keep_ratio:
            img = [
                mmcv.imrescale(im, results['scale'], return_scale=False)
                for im in results['img']
            ]
            new_h, new_w = img[0].shape[:2]
            h, w = results['img'][0].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img_0, w_scale, h_scale = mmcv.imresize(
                results['img'][0], results['scale'], return_scale=True)
            img = [img_0] + [
                mmcv.imresize(im, results['scale'], return_scale=True)[0]
                for im in results['img'][1:]
            ]
            
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        
        results['img_shape'] = img[0].shape
        results['pad_shape'] = img[0].shape
        
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

        if not is_list:
            results['img'] = results['img'][0]

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

        # print_log('resize_scale'+ str(results['scale']), logger=get_root_logger())
        for key in results.get('reg_fields', []):
            if self.keep_ratio:
                gt_offset = flow_utils.RandScale(results[key], results['scale'])
            else:
                raise NotImplementedError
            results[key] = gt_offset

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        #print_log('resize_seg', logger=get_root_logger())
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical', 'diagonal']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]
        flip = True if np.random.rand() < self.flip_ratio else False
        if 'flip' not in results:
            assert 'flip_direction' not in results
            results['flip'] = [flip]
            results['flip_direction'] = [self.direction]
        else:
            assert 'flip_direction' in results
            results['flip'].append(flip)
            results['flip_direction'].append(self.direction)
        if flip:
            # flip image
            results['img'] = [
                # use copy() to make numpy stride positive
                mmcv.imflip(im, direction=self.direction).copy()
                for im in results['img']
            ]
            
            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=self.direction).copy()
            for key in results.get('reg_fields', []):
                if (self.direction == 'horizontal'):
                    results[key] = flow_utils.RandomHorizontalFlip(results[key])
                else:
                    results[key] = flow_utils.RandomVerticalFlip(results[key])

        if not is_list:
            results['img'] = results['img'][0]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 0.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]
        if self.size is not None:
            padded_img = [
                mmcv.impad(im, shape=self.size, pad_val=self.pad_val)
                for im in results['img']
            ]
            
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    im, self.size_divisor, pad_val=self.pad_val)
                for im in results['img']
            ]
            
        results['img'] = padded_img
        results['pad_shape'] = padded_img[0].shape
        
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        if not is_list:
            results['img'] = results['img'][0]

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True,other_indexs=[]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.indexs=other_indexs

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        results['img'] = [
            mmcv.imnormalize(im, self.mean, self.std, self.to_rgb)
            for im in results['img']
        ]
        
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        if not is_list:
            results['img'] = results['img'][0]
        # print_log('normlization_finished', logger=get_root_logger())
        if len(self.indexs)!=0:
            for index in self.indexs:
                results[index]= mmcv.imnormalize(results[index], self.mean, self.std, self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class SizeConsist(object):


    def __init__(self,indexs=1,scale=4.0):

        self.indexs = indexs
        self.scale = scale

    def __call__(self, results):
        img = results['img'][self.indexs]
        img_new = np.zeros(results['img'][0].shape,np.uint8)
        img_new[:img.shape[0],:img.shape[1]] = img
        results['img'] = [results['img'][0],img_new]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, pad_vals=[(0, 0, 0), (255, 255, 255)], seg_pad_val=0):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.pad_vals = pad_vals
        assert isinstance(self.pad_vals, list)
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        img = results['img'][0]

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        # mmcv.imcrop +1 in calculate height and width
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0] - 1
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1] - 1
        crop_bbox = np.array([crop_x1, crop_y1, crop_x2, crop_y2])

        # crop the image
        results['img'] = [
            mmcv.imcrop(im, crop_bbox, pad_fill=choice(self.pad_vals))
            for im in results['img']
        ]
        results['img_shape'] = results['img'][0].shape
        

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imcrop(
                results[key], crop_bbox, pad_fill=self.seg_pad_val)
        for key in results.get('reg_fields', []):
            results[key] = mmcv.imcrop(
                results[key], crop_bbox, pad_fill=(self.seg_pad_val, self.seg_pad_val,self.seg_pad_val))

        if not is_list:
            results['img'] = results['img'][0]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        if self.ori_img_dtype == 'uint16':
            return np.clip(img, 0, 65535).astype(np.float32)
        else:
            return np.clip(img, 0, 255).astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2) and self.ori_img_dtype != 'uint16':
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2) and self.ori_img_dtype != 'uint16':
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        self.ori_img_dtype = results['ori_img_dtype']

        img = results['img']
        # random brightness
        img = [self.brightness(i) for i in img]

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = [self.contrast(i) for i in img]

        # random saturation
        img = [self.saturation(i) for i in img]

        # random hue
        img = [self.hue(i) for i in img]

        # random contrast
        if mode == 0:
            img = [self.contrast(i) for i in img]

        if not is_list:
            img = img[0]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

@PIPELINES.register_module()
class RandomChannelShiftScale(object):
    def __init__(self, max_color_shift=20, contrast_range=(0.8, 1.2)):
        assert max_color_shift >= 0
        assert len(contrast_range) == 2 
        assert all([c > 0 for c in contrast_range])
        self.max_color_shift = max_color_shift
        self.contrast_lower, self.contrast_upper = contrast_range

    def shift_scale(self, img):
        if self.max_color_shift > 0 and random.randint(2):
            C = img.shape[2]
            for c in range(C):
                shift = random.randint(-self.max_color_shift, self.max_color_shift)
                img[:,:,c] = img[:,:,c] \
                    * random.uniform(self.contrast_lower, self.contrast_upper) \
                    + shift
            if self.ori_img_dtype == 'uint16':
                return np.clip(img, 0, 65535)
            else:
                return np.clip(img, 0, 255)
        return img

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        self.ori_img_dtype = results['ori_img_dtype']
        results['img'] = [self.shift_scale(i) for i in results['img']]

        if not is_list:
            results['img'] = results['img'][0]
        return results

@PIPELINES.register_module()
class RandomGaussianBlur(object):

    def __init__(self, blur_ratio=0.5, radius=5):
        self.blur_ratio = blur_ratio
        self.radius = radius

    def gaussian_blur(self, img):
        """Brightness distortion."""
        if random.random() < self.blur_ratio:
            return cv2.GaussianBlur(img, (self.radius, self.radius), 0)
        return img

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]
        results['img'] = [self.gaussian_blur(i) for i in results['img']]

        if not is_list:
            results['img'] = results['img'][0]
        return results


@PIPELINES.register_module()
class CenterCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        rand_pair_trans_offset (float): The multi input images will have some offset to mimic the situation when images
         is not registered in building change detection
    """

    def __init__(self,
                 crop_size,
                 pad_vals=[(0, 0, 0), (255, 255, 255)],
                 seg_pad_val=0,
                 rand_pair_trans_offset=0):
        self.crop_size = crop_size
        self.pad_vals = pad_vals
        assert isinstance(self.pad_vals, list)
        self.seg_pad_val = seg_pad_val
        self.rand_pair_trans_offset = rand_pair_trans_offset

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        img = results['img'][0]
        
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = margin_h // 2
        offset_w = margin_w // 2
        # mmcv.imcrop +1 in calculate height and width
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0] - 1
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1] - 1
        crop_bbox = np.array([crop_x1, crop_y1, crop_x2, crop_y2])

        # crop the image
        for i, im in enumerate(results['img']):
            if self.rand_pair_trans_offset != 0:
                offset_h_new = offset_h + np.random.randint(
                    -1 * self.rand_pair_trans_offset,
                    self.rand_pair_trans_offset)
                offset_w_new = offset_w + np.random.randint(
                    -1 * self.rand_pair_trans_offset,
                    self.rand_pair_trans_offset)
            else:
                offset_h_new = offset_h
                offset_w_new = offset_w
            crop_y1_new, crop_y2_new = offset_h_new, offset_h_new + self.crop_size[0] - 1
            crop_x1_new, crop_x2_new = offset_w_new, offset_w_new + self.crop_size[1] - 1
            crop_bbox_new = np.array(
                [crop_x1_new, crop_y1_new, crop_x2_new, crop_y2_new])
            results['img'][i] = mmcv.imcrop(im, crop_bbox_new, pad_fill=choice(self.pad_vals))
        results['img_shape'] = results['img'][0].shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imcrop(
                results[key], crop_bbox, pad_fill=self.seg_pad_val)
        for key in results.get('reg_fields', []):
            results[key] = mmcv.imcrop(
                results[key], crop_bbox, pad_fill=(self.seg_pad_val,self.seg_pad_val, self.seg_pad_val))
        if not is_list: 
            results['img'] = results['img'][0]
        # print_log('centercrop',get_root_logger())
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate90n(object):

    def __init__(
        self,
        pad_val=0,
        seg_pad_val=0,
    ):

        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        rand_angle = random.randint(0, 4)
        if 0 == rand_angle:
            return results
        elif 1 == rand_angle:
            angle = 90
        elif 2 == rand_angle:
            angle = 180
        elif 3 == rand_angle:
            angle = 270
        else:
            raise ValueError

        results['img'] = [
            mmcv.imrotate(im, angle, border_value=self.pad_val)
            for im in results['img']
        ]
        

        # rotate semantic seg
        for key in results.get('seg_fields', []):
            # TODO mmcv doesn't have interpolate option, use cv2 for now
            # TODO: interpolate in mmcv.rotate

            results[key] = self.imrotate(
                results[key], angle, border_value=self.seg_pad_val)
        for key in results.get('reg_fields', []):
            results[key] = flow_utils.RandRotate90n(results[key], rand_angle)

        if not is_list:
            results['img'] = results['img'][0]

        return results

    @staticmethod
    def imrotate(img,
                 angle,
                 center=None,
                 scale=1.0,
                 border_value=0,
                 auto_bound=False,
                 interpolation='nearest'):
        """Rotate an image.

        Args:
            img (ndarray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple): Center of the rotation in the source image, by
                default it is the center of the image.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            auto_bound (bool): Whether to adjust the image size to cover the
                whole rotated image.

        Returns:
            ndarray: The rotated image.
        """
        interp_codes = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            borderValue=border_value,
            flags=interp_codes[interpolation])
        return rotated
@PIPELINES.register_module()
class BicubicResizeForConsist(object):

    def __call__(self,results):
        if not isinstance(results['img'], list):
            results['img'] = [results['img']]
        # hr = results['img'][1]

        temp_img = results['img'][1].transpose([-1,0,1])
        temp_img = torch.from_numpy(temp_img).float().unsqueeze(0)

        temp_img = utils_sr.interpolate(temp_img, size=(results['img'][0].shape[0], results['img'][0].shape[1]), mode='bicubic')

        temp_img = temp_img.squeeze(0)
        temp_img = temp_img.numpy().transpose([1, 2, 0]).astype(np.uint8)
        results['img']=[results['img'][0],temp_img]

        return results
@PIPELINES.register_module()
class Copy3Img(object):

    def __call__(self,results):
        results['img']=[results['img'][0],results['img'][1],results['img'][1]]

        return results
@PIPELINES.register_module()
class RandImgList(object):

    def __call__(self,results):
        if random.random()>0.5:
            results['img']=[results['img'][1],results['img'][0]]

        return results
@PIPELINES.register_module()
class RandomYoco(object):
    def __init__(self,
                 rate=0.5,
                 ):
        self.rate=rate
    def __call__(self,results):
        if torch.rand(1) > self.rate:
            return results

        if not isinstance(results['img'], list):
            results['img'] = [results['img']]
        _,h,w = np.shape(results['img'][0])
        if torch.rand(1) > 0.5:
            img_temp0 = np.concatenate([results['img'][0][:,:,:int(w/2)],results['img'][1][:,:,int(w/2):]],2)
            img_temp1 = np.concatenate([results['img'][1][:, :, :int(w / 2)], results['img'][0][:, :, int(w / 2):]], 2)
        else:
            img_temp0 = np.concatenate([results['img'][0][:, :int(h / 2), :], results['img'][1][:, int(h/2):, :]], 1)
            img_temp1 = np.concatenate([results['img'][1][:, :int(h / 2), :], results['img'][0][:, int(h/2):, :]], 1)

        results['img']=[img_temp0,img_temp1]

        return results


@PIPELINES.register_module()
class RandomGaussianBlurNosie(object):
    def __init__(self,
                 scale=4,
                 sig_min=0.2,
                 sig_max=4.0,
                 random_kernel=True,
                 ksize=21,
                 noise_high=0,
                 noise=False,
                 rate_cln=1.0,
                 rate_iso=1.0,
                 random_disturb=False,
                ):
        self.SRMDPreprocessing= utils_sr.SRMDPreprocessing(
            scale=scale,
            noise=noise,
            noise_high=noise_high,
            sig_min=sig_min,
            sig_max=sig_max,
            rate_cln = rate_cln,
            random_kernel=random_kernel,
            ksize=ksize,
            rate_iso=rate_iso,
            random_disturb=random_disturb,
            return_hr_bicubic_from_lr=True
        )

    def __call__(self, results):

        hr = results['img'][1]

        hr_tensor = utils_sr.img2tensor(hr)
        C, H, W = hr_tensor.size()
        hr_tensor = hr_tensor.view(1,C,H,W)

        lr_tensor=self.SRMDPreprocessing(hr_tensor)
        lr_blur_bicubic= utils_sr.tensor2img(lr_tensor)

        results['img'] = [results['img'][0], lr_blur_bicubic]

        return results
@PIPELINES.register_module()
class RandomGaussianBlurNosieBicubicSR(object):
    def __init__(self,
                 scale=4,
                 sig_min=0.2,
                 sig_max=4.0,
                 random_kernel=True,
                 ksize=21,
                 noise_high=0,
                 noise=False,
                 rate_cln=1.0,
                 rate_iso=1.0,
                 random_disturb=False,
                ):
        self.SRMDPreprocessing= utils_sr.SRMDPreprocessing(
            scale=scale,
            noise=noise,
            noise_high=noise_high,
            sig_min=sig_min,
            sig_max=sig_max,
            rate_cln = rate_cln,
            random_kernel=random_kernel,
            ksize=ksize,
            rate_iso=rate_iso,
            random_disturb=random_disturb,
            return_hr_bicubic_from_lr=True
        )

    def __call__(self, results):

        hr = results['img'][1]

        hr_tensor = utils_sr.img2tensor(hr)
        C, H, W = hr_tensor.size()
        hr_tensor = hr_tensor.view(1,C,H,W)

        lr_tensor=self.SRMDPreprocessing(hr_tensor)
        lr_blur_bicubic= utils_sr.tensor2img(lr_tensor)

        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], lr_blur_bicubic]

        return results
@PIPELINES.register_module()
class RandomGaussianBlurNosieSR(object):
    def __init__(self,
                 scale=4,
                 sig_min=0.2,
                 sig_max=4.0,
                 random_kernel=True,
                 ksize=21,
                 noise_high=0,
                 noise=False,
                 rate_cln=1.0,
                 rate_iso=1.0,
                 random_disturb=False,
                ):
        self.SRMDPreprocessing= utils_sr.SRMDPreprocessing(
            scale=scale,
            noise=noise,
            noise_high=noise_high,
            sig_min=sig_min,
            sig_max=sig_max,
            rate_cln = rate_cln,
            random_kernel=random_kernel,
            ksize=ksize,
            rate_iso=rate_iso,
            random_disturb=random_disturb
        )

    def __call__(self, results):

        hr = results['img'][1]

        hr_tensor = utils_sr.img2tensor(hr)
        C, H, W = hr_tensor.size()
        hr_tensor = hr_tensor.view(1,C,H,W)

        lr_tensor,kernel,lr_blured_t,lr_t =self.SRMDPreprocessing(hr_tensor)
        B2, C2, H2, W2 = lr_tensor.size()
        lr = utils_sr.tensor2img(lr_tensor)
        image_kernel = utils_sr.tensor2img(kernel)
        lr_blured_bicubic = utils_sr.tensor2img(lr_blured_t)
        lr_bicubic = utils_sr.tensor2img(lr_t)

        lr_temp = np.zeros([H,W,C],np.uint8)
        lr_temp[:H2,:W2,:] = lr
        lr_no_noise=np.concatenate([lr_blured_bicubic,lr_bicubic],-1).astype(np.float32) / 255.0

        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], lr_temp]
        results['kernel'] = image_kernel
        results['lr_wo_noise']= lr_no_noise


        return results


@PIPELINES.register_module()
class RandomDegForSR(object):
    def __init__(self,noise_level=15,scales = [1,1/4]):
        self.scale_list = scales
        self.noise_level =noise_level
    def __call__(self, results):

        hr = results['img'][1]

        H, W, _ = hr.shape
        l_size = random.uniform(self.scale_list[0],self.scale_list[1])
        img_deg = utils_sr.random_deg(hr, l_size, noise_level=self.noise_level)

        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], img_deg, hr]
        return results

@PIPELINES.register_module()
class RandomDegForSRv2(object):
    def __init__(self,noise_level=15,scales = [1,1/4]):
        self.scale_list = scales
        self.noise_level =noise_level
    def __call__(self, results):

        hr = results['img'][1]

        H, W, _ = hr.shape
        l_size = random.uniform(self.scale_list[0],self.scale_list[1])
        img_deg = utils_sr.random_deg(hr, l_size, noise_level=self.noise_level)

        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], img_deg]

        return results

@PIPELINES.register_module()
class RandomDegForSRv3(object):
    def __init__(self,noise_level=15,scales = [1,1/4]):
        self.scale_list = scales
        self.noise_level =noise_level
    def __call__(self, results):

        hr = results['img'][1]

        H, W, _ = hr.shape
        l_size = random.uniform(self.scale_list[0],self.scale_list[1])
        img_deg = utils_sr.random_degwoup(hr, l_size, noise_level=self.noise_level)

        H2, W2, C2 = img_deg.shape
        lr_temp = np.zeros(hr.shape, np.uint8)
        lr_temp[:H2, :W2, :] = img_deg

        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], lr_temp]
        return results

@PIPELINES.register_module()
class RandBicubicResize(object):
    def __init__(self, scales=[1/8, 1 / 4]):
        self.scale_list = scales
    def __call__(self,results):
        if not isinstance(results['img'], list):
            results['img'] = [results['img']]

        l_size = random.uniform(self.scale_list[0], self.scale_list[1])

        temp_img = results['img'][1].transpose([-1,0,1])
        temp_img = torch.from_numpy(temp_img).float().unsqueeze(0)

        temp_img = utils_sr.interpolate(temp_img, scale_factor=(l_size, l_size), mode='bicubic')

        temp_img = utils_sr.interpolate(temp_img, size=(results['img'][0].shape[0], results['img'][0].shape[1]), mode='bicubic')

        temp_img = temp_img.squeeze(0)
        temp_img = temp_img.numpy().transpose([1, 2, 0]).astype(np.uint8)
        results['img']=[results['img'][0],temp_img]

        return results

@PIPELINES.register_module()
class GetHrForSR(object):
    def __init__(self,index=1):
        self.index=index
    def __call__(self, results):
        hr = results['img'][self.index]
        results['hr'] = hr.copy()
        return results



@PIPELINES.register_module()
class BicubicSR(object):
    def __init__(self,scale):
        self.scale = scale

    def __call__(self, results):

        hr = results['img'][1]
        lr= utils_sr.imresize(hr, self.scale)
        H2, W2, C2 = lr.shape
        lr_temp = np.zeros(hr.shape,np.uint8)
        lr_temp[:H2,:W2,:] = lr
        results['hr'] = hr.copy()
        results['img'] = [results['img'][0], lr_temp]

        return results

@PIPELINES.register_module()
class RandomRotate(object):
    """Random rotate the image & seg.

    Args:
        rotate_range (tuple): Expected range for rotation (min, max).
        rotate_ratio (float, optional): The rotation probability.
    """

    def __init__(self,
                 rotate_range,
                 pad_vals=[(0, 0, 0), (255, 255, 255)],
                 seg_pad_val=0,
                 rotate_ratio=None):
        self.rotate_range = rotate_range
        self.pad_vals = pad_vals
        assert isinstance(self.pad_vals, list)
        self.seg_pad_val = seg_pad_val
        if rotate_ratio is not None:
            assert rotate_ratio >= 0 and rotate_ratio <= 1
        self.rotate_ratio = rotate_ratio

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        if self.rotate_ratio is None or np.random.rand() < self.rotate_ratio:
            angle = np.random.uniform(*self.rotate_range)
            # random padding 0 or 255
            results['img'] = [
                mmcv.imrotate(im, angle, border_value = choice(self.pad_vals))
                for im in results['img']
            ]

            # rotate semantic seg
            for key in results.get('seg_fields', []):
                # TODO mmcv doesn't have interpolate option, use cv2 for now
                # TODO: interpolate in mmcv.rotate
                results[key] = self.imrotate(results[key], angle, border_value=self.seg_pad_val)
            for key in results.get('reg_fields', []):
                results[key] = flow_utils.RandRotate(results[key], 360 - angle)
        if not is_list:
            results['img'] = results['img'][0]


        return results

    @staticmethod
    def imrotate(img,
                 angle,
                 center=None,
                 scale=1.0,
                 border_value=0,
                 auto_bound=False,
                 interpolation='nearest'):
        """Rotate an image.

        Args:
            img (ndarray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple): Center of the rotation in the source image, by
                default it is the center of the image.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            auto_bound (bool): Whether to adjust the image size to cover the
                whole rotated image.

        Returns:
            ndarray: The rotated image.
        """
        interp_codes = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            borderValue=border_value,
            flags=interp_codes[interpolation])
        return rotated

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(rotate_range={self.rotate_range}, rotate_ratio=' \
               f'{self.rotate_ratio})'

# @PIPELINES.register_module()
# class Label1234ToLabel12345678(object):
#     def __init__(self):
#         self.transform = seg_transforms.label1234_2_label12345678()
#     def __call__(self, results):
#
#         label_roofside_origin, label_roofside_5edge= self.transform(results[results['gt_semantic_seg']])
#         results['label_seg'] = label_roofside_5edge
#         # results['label_roofside_origin']=label_roofside_origin
#         return results
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str

@PIPELINES.register_module()
class GenerateMultiSegLabels(object):
    def __init__(self,wmap_vec=[1,1,1,1,1], vertex_wmap=1):
        self.transform1 = seg_utils.label1234_2_label12345678()
        self.transform2 = seg_utils.label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls()
        self.transform3 = seg_utils.label_roof_side_to_edgeorient(wmap_vec=wmap_vec, vertex_wmap=vertex_wmap)
    def __call__(self, results):
        label_roofside_origin, label_roofside_5edge = self.transform1(results['gt_semantic_seg'])
        label_roof,label_roofside,label_edge5cls,label_side= self.transform2(label_roofside_5edge)
        label_edge_orient = self.transform3(label_roof,label_side,label_roofside_5edge)
        results['gt_semantic_seg'] = label_roofside
        results['gt_edge_seg'] = label_edge5cls
        results['gt_orient_edge_36_seg'] = label_edge_orient
        results['seg_fields'].append('gt_edge_seg')
        results['seg_fields'].append('gt_orient_edge_36_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomPixelNoise(object):
    """ Add pixel noise to image

    Args:
        max_pixel_noise (float, optional): The max value of pixel noise. Default: 10.
        rand_max (bool, optional): Randomly choose the maximum of the pixel noise. 
            Default: True.
    """

    def __init__(self,
                 max_pixel_noise=10,
                 rand_max=True):
        assert max_pixel_noise >= 0
        self.max_pixel_noise = max_pixel_noise
        self.rand_max = rand_max

    def pixel_noise(self, img):
        if self.rand_max:
            noise = np.random.randint(
                random.randint(-1*self.max_pixel_noise, 0),
                random.randint(1, self.max_pixel_noise),
                size=img.shape,
                dtype='int')
        else:
            noise = np.random.randint(
                -1 * self.max_pixel_noise,
                self.max_pixel_noise,
                size=img.shape,
                dtype='int')
        return img.astype(np.uint8) + noise

    def __call__(self, results):
        if isinstance(results['img'], list):
            is_list = True
        else:
            is_list = False
            results['img'] = [results['img']]

        if results['ori_img_dtype'] == 'uint16':
            results['img'] = [np.clip(self.pixel_noise(i), 0, 65535).astype(np.float32)
                for i in results['img']]
        else:
            results['img'] = [np.clip(self.pixel_noise(i), 0, 255).astype(np.uint8)
                for i in results['img']]
        
        if not is_list:
            results['img'] = results['img'][0]
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class FlowConvert(object):
    """ Change Flow_reg from (3,H,W) to (2,H,W),change
            ignore value  to nan from(+-400),(+-500),(255)
        Input ：numpy array (3,H,W)
        Output:numpy array (2,H,W)
    Args:
        value_threshold (float, optional): The maximum threshold of pixels to be
            replaced. Default: 400.
        value_fill (np.dtype, optional): Value after replacement. Default: np.nan.
    """

    def __init__(self,keys=[],
                 value_threshold=400,
                 value_fill=np.nan):

        self.value_threshold = value_threshold
        self.value_fill = value_fill
        self.keys=keys

    def __call__(self, results):
        for key in self.keys:
            assert key in results

            flow_reg = results[key]
            flow_reg_2c = flow_reg[:, :, 0:2].copy()  # (2,H,W)
            flow_reg_2c[np.where(
                np.abs(flow_reg) >= self.value_threshold)] = self.value_fill  # change the ignore value to np.nan
            ignore_255 = np.where(np.abs(flow_reg[:, :, 2]) == 255)  # get where the channel 3 is 255
            flow_reg_2c[:, :, 0][ignore_255] = self.value_fill
            flow_reg_2c[:, :, 1][ignore_255] = self.value_fill
            results[key] = flow_reg_2c

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class CatLPIPSWithImages(object):
    def __call__(self, results):
        # raise results['gt_lpips'].shape
        results['img'].append(results['gt_lpips'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
@PIPELINES.register_module()
class CatBuildSegLabelsWithImages(object):
    def __call__(self, results):
        # raise results['gt_lpips'].shape
        results['img'].append(results['gt_build_seg_1'])
        results['img'].append(results['gt_build_seg_2'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str