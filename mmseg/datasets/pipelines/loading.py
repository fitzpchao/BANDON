import os
import os.path as osp

import mmcv
import numpy as np
import cv2

import rasterio
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load images from files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 bgrn2bgr=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filenames = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filenames = results['img_info']['filename']

        if isinstance(filenames, list):
            is_list = True
        else:
            is_list = False
            filenames = [filenames]
        imgs = []
        for filename in filenames:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)

            ori_img_dtype = img.dtype
            if self.to_float32 or ori_img_dtype == 'uint16':
                img = img.astype(np.float32)
            imgs.append(img)

        img_shape = imgs[0].shape
        # if is_list:
        #     for img in imgs[1:]:
        #         assert img_shape == img.shape

        results['filename'] = filenames
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = imgs if is_list else imgs[0]
        results['img_shape'] = img_shape
        results['ori_shape'] = img_shape
        results['ori_img_dtype'] = ori_img_dtype
        # Set initial values for default meta_keys
        results['pad_shape'] = img_shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img_shape) < 3 else imgs[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class LoadAnnotationsWithMultiLabels(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow',
                 in_keys=['']):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.in_keys=in_keys
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
            filename_seg_build = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_build'])
            filename_flow_build = osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_build'])
            filename_flow_cd= osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_cd'])


        else:
            filename = results['ann_info']['seg_map']
            if 'seg_build' in results['ann_info'].keys():
                filename_seg_build = results['ann_info']['seg_build']

            if 'flow_build' in results['ann_info'].keys():
                filename_flow_build = results['ann_info']['flow_build']
            if 'flow_cd' in results['ann_info'].keys():
                filename_flow_cd = results['ann_info']['flow_cd']

        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)



        if 'seg_build' in results['ann_info'].keys() and 'seg_build' in self.in_keys:
            gt_build_seg_1 = mmcv.imfrombytes(
                self.file_client.get(filename_seg_build[0]), flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            gt_build_seg_2 = mmcv.imfrombytes(
                self.file_client.get(filename_seg_build[1]), flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            results['gt_build_seg_1'] = gt_build_seg_1
            results['gt_build_seg_2'] = gt_build_seg_2
            results['seg_fields'].append('gt_build_seg_1')
            results['seg_fields'].append('gt_build_seg_2')

        if 'flow_build' in results['ann_info'].keys() and 'flow_build' in self.in_keys :

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(filename_flow_build[0]) as src:
                    gt_flow_build_reg_1 = src.read()
                    gt_flow_build_reg_1 = np.concatenate(
                        [gt_flow_build_reg_1,
                         np.zeros([1, gt_flow_build_reg_1.shape[1], gt_flow_build_reg_1.shape[2]])], 0)
                    gt_flow_build_reg_1 = np.transpose(gt_flow_build_reg_1, [1, 2, 0])
                    gt_flow_build_reg_1 = cv2.resize(gt_flow_build_reg_1, (2048, 2048), interpolation=cv2.INTER_LINEAR)

                with rasterio.open(filename_flow_build[1]) as src:
                    gt_flow_build_reg_2 = src.read()
                    gt_flow_build_reg_2 = np.concatenate([gt_flow_build_reg_2, np.zeros(
                        [1, gt_flow_build_reg_2.shape[1], gt_flow_build_reg_2.shape[2]])], 0)
                    gt_flow_build_reg_2 = np.transpose(gt_flow_build_reg_2, [1, 2, 0])
                    gt_flow_build_reg_2 = cv2.resize(gt_flow_build_reg_2, (2048, 2048), interpolation=cv2.INTER_LINEAR)

            results['gt_flow_build_reg_1'] = gt_flow_build_reg_1
            results['gt_flow_build_reg_2'] = gt_flow_build_reg_2
            results['reg_fields'].append('gt_flow_build_reg_1')
            results['reg_fields'].append('gt_flow_build_reg_2')

        if 'flow_cd' in results['ann_info'].keys() and 'flow_cd' in self.in_keys:

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(filename_flow_cd) as src:
                    gt_flow_cd_reg = src.read()
                    gt_flow_cd_reg = np.transpose(gt_flow_cd_reg, [1, 2, 0])
                    ignore = gt_flow_cd_reg[..., -1] == 255

                    gt_flow_cd_reg[..., -1][ignore] = 0
                    gt_flow_cd_reg = cv2.resize(gt_flow_cd_reg, (2048, 2048), interpolation=cv2.INTER_LINEAR)
                    mask_cd = gt_flow_cd_reg[..., -1] > 0
                    gt_flow_cd_reg[..., -1][mask_cd] = 255
            results['gt_flow_cd_reg'] = gt_flow_cd_reg
            results['reg_fields'].append('gt_flow_cd_reg')

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg



        results['seg_fields'].append('gt_semantic_seg')





        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str








@PIPELINES.register_module()
class LoadAnnotationsWithSegLabels(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
            filename_seg_build = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_build'])
            filename_flow_build = osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_build'])
            filename_flow_cd= osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_cd'])


        else:
            filename = results['ann_info']['seg_map']
            filename_seg_build = results['ann_info']['seg_build']
            filename_flow_build = results['ann_info']['flow_build']
            filename_flow_cd = results['ann_info']['flow_cd']

        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # gt_flow_cd_reg = np.load(filename_flow_cd)
        # gt_flow_build_reg_1 = np.load(filename_flow_build[0])
        # gt_flow_build_reg_2 = np.load(filename_flow_build[1])


        gt_build_seg_1 = mmcv.imfrombytes(
                self.file_client.get(filename_seg_build[0]), flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_build_seg_2 = mmcv.imfrombytes(
            self.file_client.get(filename_seg_build[1]), flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)



        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        # results['gt_flow_cd_reg'] = gt_flow_cd_reg
        results['gt_build_seg_1'] = gt_build_seg_1
        results['gt_build_seg_2'] = gt_build_seg_2

        results['seg_fields'].append('gt_semantic_seg')
        results['seg_fields'].append('gt_build_seg_1')
        results['seg_fields'].append('gt_build_seg_2')


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsWithOffsetLabels(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
            filename_seg_build = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_build'])
            filename_flow_build = osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_build'])
            filename_flow_cd= osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_cd'])


        else:
            filename = results['ann_info']['seg_map']
            filename_seg_build = results['ann_info']['seg_build']
            filename_flow_build = results['ann_info']['flow_build']
            filename_flow_cd = results['ann_info']['flow_cd']

        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        gt_flow_build_reg_1 = np.load(filename_flow_build[0])
        # gt_flow_build_reg_1 = np.load(filename_flow_cd)
        gt_flow_build_reg_2 = np.load(filename_flow_build[1])




        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['gt_flow_build_reg_1'] = gt_flow_build_reg_1
        results['gt_flow_build_reg_2'] = gt_flow_build_reg_2

        results['seg_fields'].append('gt_semantic_seg')
        results['reg_fields'].append('gt_flow_build_reg_1')
        results['reg_fields'].append('gt_flow_build_reg_2')


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotationsWithFlowCDLabels(object):
    """Load annotations for semantic segmentation.

    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
            filename_seg_build = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_build'])
            filename_flow_build = osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_build'])
            filename_flow_cd= osp.join(results['seg_prefix'],
                                          results['ann_info']['flow_cd'])


        else:
            filename = results['ann_info']['seg_map']
            filename_seg_build = results['ann_info']['seg_build']
            filename_flow_build = results['ann_info']['flow_build']
            filename_flow_cd = results['ann_info']['flow_cd']

        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        gt_flow_cd_reg = np.load(filename_flow_cd)

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['gt_flow_cd_reg'] = gt_flow_cd_reg

        results['seg_fields'].append('gt_semantic_seg')
        results['reg_fields'].append('gt_flow_cd_reg')



        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str