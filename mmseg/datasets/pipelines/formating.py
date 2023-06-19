from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [to_tensor(tmp) for tmp in results[key]]
                results[key] = np.concatenate(results[key], axis=2)
            else:
                results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if not isinstance(img, list):
                img = [img]
            for i in range(len(img)):
                if len(img[i].shape) < 3:
                    img[i] = np.expand_dims(img[i], -1)
                img[i] = img[i].transpose(2, 0, 1)
                img[i] = to_tensor(img[i])
            results[key] = np.concatenate(img, axis=0)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Transpose(object):
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer(object):
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True),
            dict(key='gt_semantic_seg'))``.
    """

    def __init__(self,
                 fields=(dict(key='img',
                              stack=True), dict(key='gt_semantic_seg'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - fields in 'seg_fields', e.g. gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    - fields in 'reg_fields': (1)to tensor, (2)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if not isinstance(img, list):
                img = [img]
            for i in range(len(img)):
                if len(img[i].shape) < 3:
                    img[i] = np.expand_dims(img[i], -1)
                img[i] = np.ascontiguousarray(img[i].transpose(2, 0, 1))
            img = np.concatenate(img, axis=0)
            results['img'] = DC(to_tensor(img), stack=True)
        for key in results.get('seg_fields', []):
            # convert to long
            if('gt_edge_seg' in key):
                results[key] = DC(to_tensor(results[key][None,...].astype(np.int64)), stack=True)
            elif('gt_orient_edge_36_seg' in key):
                results[key] = DC(to_tensor(results[key][None,...].astype(np.int64)), stack=True)
            else:
                results[key] = DC(to_tensor(results[key][None,
                                                         ...].astype(np.int64)),
                                  stack=True)

        for key in results.get('reg_fields', []):
            # convert to float32
            if 'flow' in key:
                label_offset = results[key]
                label_offset = np.ascontiguousarray(label_offset)
                label_offset = torch.from_numpy(label_offset)
                label_offset = label_offset.permute(2, 0, 1)
                if '_x' in key:
                    raise Exception(key,label_offset.size())
                if not isinstance(label_offset, torch.FloatTensor):
                    label_offset = label_offset.float()
                results[key] = DC(label_offset, stack=True)
            else:
                results[key] = DC(to_tensor(results[key].astype(np.float32)),
                                stack=True)

        if 'patch_label' in results:
            if len(results['patch_label']['gt_semantic_seg']) == 1:
                results['patch_label']['gt_semantic_seg'] = DC(
                    to_tensor(
                        results['patch_label']['gt_semantic_seg'][0][None, ...]),
                    stack=True)
            else:
                results['patch_label']['gt_semantic_seg'] = DC(
                    to_tensor(
                        np.stack(results['patch_label']['gt_semantic_seg'])),
                    stack=True)
        if 'weightmaps' in results:
            for key in results['weightmaps']:
                results['weightmaps'][key] = DC(
                    to_tensor(results['weightmaps'][key][None, ...]),
                    stack=True)

        if 'hr' in results:
            img = results['hr']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['hr'] = DC(to_tensor(img), stack=True)

        if 'kernel' in results:
            img = results['kernel']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['kernel'] = DC(to_tensor(img), stack=True)

        if 'lr_wo_noise' in results:
            img = results['lr_wo_noise']
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['lr_wo_noise'] = DC(to_tensor(img), stack=True)

        if 'gt_labels' in results:
            labels = results['gt_labels']
            results['gt_labels'] = DC(labels)

        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 #
                 meta_keys=('filename', 'ori_filename', 'ori_shape','flip','flip_direction',
                            'img_shape', 'pad_shape', 'scale_factor',  'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
            # print_log('key,size:'+ str(data[key].size()),get_root_logger())

        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
