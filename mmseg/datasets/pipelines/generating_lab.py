import mmcv
import numpy as np
from numpy import random

from skimage.segmentation import clear_border, find_boundaries
from skimage.morphology import (remove_small_objects, binary_dilation,
    binary_opening, disk)
from skimage.filters import gaussian, sobel_h, sobel_v
from scipy import ndimage as ndi

from .orientation_util import OrientationUtil 
from ..builder import PIPELINES
import torch
from sklearn import neighbors

@PIPELINES.register_module()
class LPIPS2Weights(object):
    def __init__(self,
                 src_key='gt_semantic_seg'):
        self.src_key = src_key

    def __call__(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Decoded results.
        """
        assert self.src_key in results


        if 'weightmaps' not in results:
            results['weightmaps'] = {}

        dst_weightmap = results[self.src_key].copy()
        min_v = np.min(dst_weightmap)
        max_v = np.max(dst_weightmap)

        dst_weightmap = (dst_weightmap - min_v)/(max_v-min_v + 1e-6)*5.0


        results['weightmaps'][self.src_key] = dst_weightmap

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class LPIPS2WeightsGaussian(object):
    def __init__(self,
                 src_key='gt_semantic_seg'):
        self.src_key = src_key

    def __call__(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Decoded results.
        """
        assert self.src_key in results

        if 'weightmaps' not in results:
            results['weightmaps'] = {}

        dst_weightmap = results[self.src_key].copy()
        mean = 0.53
        std = 0.15
        dst_weightmap = np.clip(dst_weightmap,0,mean + 3*std)*5.0



        results['weightmaps'][self.src_key] = dst_weightmap

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class DecodeLabel(object):
    """ decode gt map 'src_key' to multiple gt maps and corresponding
        weightmaps
    Args:
        src_key (str, optional): The key of the gt map to be deocded.
            Default: 'gt_semantic_seg'.
        dst_keys (list[str], optional): The keys of decoded gt maps.
            Default: ['gt_semantic_seg'].
        dst_values (list[list[int]]]): The decoded values of gt maps.
        dst_weightmap_values (list[list[float]] | None, optional): The decoded value 
            of weightmaps.
    """

    def __init__(self,
                 src_key='gt_semantic_seg',
                 dst_keys=['gt_semantic_seg'],
                 dst_values=None,
                 dst_weightmap_values=None,
                 ignore_index=255):
        assert isinstance(dst_keys, list)
        assert isinstance(dst_values, list)
        assert len(dst_keys) == len(dst_values)
        dst_values_len = set([len(v) for v in dst_values])
        assert len(dst_values_len) == 1
        if dst_weightmap_values is not None:
            assert isinstance(dst_weightmap_values, list)
            assert len(dst_keys) == len(dst_weightmap_values)
            dst_weightmap_values_len = set([len(v) for v in dst_weightmap_values
                if v is not None])
            assert len(dst_weightmap_values_len) == 1
            assert dst_weightmap_values_len == dst_values_len 

        self.src_key = src_key
        self.dst_keys = dst_keys
        self.dst_values = dst_values
        self.dst_weightmap_values = dst_weightmap_values 
        self.ignore_index = ignore_index
        self.src_unq_len = list(dst_values_len)[0]
    def __call__(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Decoded results. 
        """
        assert self.src_key in results

        if self.dst_weightmap_values is not None:
            if 'weightmaps' not in results:
                results['weightmaps'] = {}

        src_label = results[self.src_key].copy()
        unq_src = set(np.unique(src_label))
        assert unq_src.issubset(set(list(range(self.src_unq_len))+[255]))

        for i in range(len(self.dst_keys)):
            dst_label = np.ones_like(src_label) * self.ignore_index
            for v in range(self.src_unq_len):
                dst_label[src_label == v] = self.dst_values[i][v]
            dst_label[src_label == self.ignore_index] = self.ignore_index

            results[self.dst_keys[i]] = dst_label
            if self.dst_keys[i] not in results['seg_fields']:
                results['seg_fields'].append(self.dst_keys[i])

            if self.dst_weightmap_values is not None and \
                self.dst_weightmap_values[i] is not None :
                dst_weightmap = np.zeros_like(src_label, dtype=np.float32)
                for v in range(self.src_unq_len):
                    dst_weightmap[src_label == v] = self.dst_weightmap_values[i][v]
                dst_weightmap[src_label == self.ignore_index] = 0.0

                results['weightmaps'][self.dst_keys[i]] = dst_weightmap

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str


@PIPELINES.register_module()
class Reg2ClsXY(object):
    def __call__(self, results):

        def reg2cls_flow(array, ths):
            masks = []
            for i, th in enumerate(ths):
                if i == 0:
                    masks.append(array < th)
                    masks.append(np.logical_and(array > th, array < ths[i + 1]))
                elif (i == (len(ths) - 1)):
                    masks.append(array > th)
                else:
                    temp_mask = np.logical_and(array > th, array < ths[i + 1])
                    masks.append(temp_mask)
            return masks

        for key in results.get('reg_fields', []):
            flow = results[key]
            x, y = flow[:, :, 0], flow[:, :, 1]
            ig_area_x = np.abs(x) >= 400
            ig_area_y = np.abs(y) >= 400
            ig_area = np.logical_or(ig_area_x,ig_area_y)
            ths = [-10, 0, 10]
            masks_x = reg2cls_flow(x, ths)
            masks_y = reg2cls_flow(y, ths)
            cls_x = np.zeros(x.shape, np.uint8)
            cls_y = np.zeros(x.shape, np.uint8)
            for i, mask in enumerate(masks_x):
                cls_x[mask] = i + 1
                cls_y[masks_y[i]] = i + 1
            cls_x[x == 0] = 0
            cls_y[y == 0] = 0

            ig_mask = flow[:, :, 2] == 255
            cls_x[ig_mask] = 255
            cls_y[ig_mask] = 255
            cls_x[ig_area] = 255
            cls_y[ig_area] = 255


            results[key + '_x'] = cls_x
            results[key + '_y'] = cls_y
            results['seg_fields'].append(key + '_x')
            results['seg_fields'].append(key + '_y')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class Reg2ClsXYWithSeg(object):
    def __init__(self,ths=[-10,0,10],in_keys=[]):
        self.ths=ths
        self.inkeys=in_keys
    def __call__(self, results):

        def reg2cls_flow(array, ths):
            masks = []
            for i, th in enumerate(ths):
                if i == 0:
                    masks.append(array < th)
                    masks.append(np.logical_and(array > th, array < ths[i + 1]))
                elif (i == (len(ths) - 1)):
                    masks.append(array > th)
                else:
                    temp_mask = np.logical_and(array > th, array < ths[i + 1])
                    masks.append(temp_mask)
            return masks

        for key in self.inkeys:
            flow = results[key]
            x, y = flow[:, :, 0], flow[:, :, 1]
            ig_area_x = np.abs(x) >= 400
            ig_area_y = np.abs(y) >= 400
            ig_area = np.logical_or(ig_area_x, ig_area_y)
            ths = self.ths
            masks_x = reg2cls_flow(x, ths)
            masks_y = reg2cls_flow(y, ths)
            cls_x = np.zeros(x.shape, np.uint8)
            cls_y = np.zeros(x.shape, np.uint8)
            for i, mask in enumerate(masks_x):
                cls_x[mask] = i + 2
                cls_y[masks_y[i]] = i + 2
            if '1' in key:
                seg = results['gt_build_seg_1']
            elif '2' in key:
                seg = results['gt_build_seg_2']
            else:
                seg = results['gt_build_seg_1']

            cls_x[seg == 0] = 0
            cls_x[np.logical_and(seg != 0, x == 0)] = 1

            cls_y[seg == 0] = 0
            cls_y[np.logical_and(seg != 0, y == 0)] = 1


            ig_mask = flow[:, :, 2] == 255
            cls_x[ig_mask] = 255
            cls_y[ig_mask] = 255
            cls_x[ig_area] = 255
            cls_y[ig_area] = 255

            results[key + '_x'] = cls_x
            results[key + '_y'] = cls_y
            results['seg_fields'].append(key + '_x')
            results['seg_fields'].append(key + '_y')

        return results



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class Label2OneHot(object):
    def __init__(self, keys=['']):
        self.keys = keys
    def __call__(self, results):
        for key in self.keys:
            temp = results[key].copy()
            temp[temp==255]=3
            temp = torch.from_numpy(temp).long().unsqueeze(0)
            gt_onehot = torch.zeros((4, temp.shape[1], temp.shape[2]))
            gt_onehot.scatter_(0, temp, 1).float()
            results[key] =  np.transpose(gt_onehot.numpy(),[1,2,0])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str

@PIPELINES.register_module()
class FlowIgnoreChangeArea(object):
    def __init__(self, keys=['']):

        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            temp = results[key].copy()
            mask = results['gt_semantic_seg'] == 1
            temp[mask]=255
            results[key] = temp


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class FlowRegIgnoreChangeArea(object):
    def __init__(self, keys=['']):

        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            temp = results[key].copy()
            mask = results['gt_semantic_seg'] == 1
            temp[mask,-1]=255
            results[key] = temp


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str

@PIPELINES.register_module()
class Seg2LabelMask(object):
    def __init__(self,ignore_label=255):
        self.ignore_label = ignore_label
    def __call__(self, results):
        seg = results['gt_semantic_seg'].copy()
        classes = np.unique(seg)
        # raise Exception(seg.size())
        h,w = seg.shape[-2:]

        # remove ignored region
        classes = classes[classes != self.ignore_label]
        gt_classes = torch.tensor(classes, dtype=torch.int64)
        masks = []
        for class_id in classes:
            mask = seg == class_id
            masks.append(mask.astype(np.uint8))

        from mmseg.models.decode_heads.mask2former_utils.structures import BitmapMasks
        results["gt_masks"] = BitmapMasks(masks,h,w)
        results["gt_labels"] = gt_classes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class Seg2LabelMaskInstance(object):
    def __init__(self,ignore_label=255):
        self.ignore_label = ignore_label
    def __call__(self, results):

        import cv2

        def raster2poly(label_img):
            """Convert binary mask to shapely polygon

            Args:
                label_img (ndarry): bianry mask image np array (N x H x 1)that need to be converted
                contours_img_fn (str): filename of the output conoutrs image
                fg_value (int): the value that represents foreground pixel
                # debug (bool, optional): [description]. Defaults to False.

            Returns:
                polygons , conoturs: shapely polygons and contours of the instance level mask in label_image
            """
            if len(label_img.shape) == 2:
                label_img = label_img[:, :, None]
            label_bi = label_img.astype(np.uint8)
            contours, hierarchy_list = cv2.findContours(
                label_bi, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
            hierarchy = hierarchy_list[0] if hierarchy_list is not None else []
            n_contours = len(contours)
            contours_used_flag = [False] * n_contours
            polygons = []
            ext_contours = []
            int_contours = []
            for i in range(n_contours):
                if contours_used_flag[i]:
                    continue
                contours_used_flag[i] = True
                i_contour = contours[i]
                exterior = i_contour[:, 0, :]
                ext_contours.append(i_contour)
                First_child_index = hierarchy[i][2]
                if len(exterior) <= 2:
                    continue
                polygons.append(exterior)

            return polygons
        seg = results['gt_semantic_seg'].copy()
        h,w = seg.shape[-2:]
        mask = seg > 0
        masks = []
        ps = raster2poly(mask)
        classes=[]
        # classes.append(0)
        # masks.append((seg==0).astype(np.uint8))
        for i in range(len(ps)):
            temp = np.zeros([h, w], np.uint8)
            temp = cv2.drawContours(temp, [ps[i]], -1, 1, cv2.FILLED)
            masks.append(temp)
            classes.append(0)
        # raise Exception(seg.size())


        # remove ignored region

        gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)

        from mmseg.models.decode_heads.mask2former_utils.structures import BitmapMasks
        results["gt_masks"] = BitmapMasks(masks,h,w)
        results["gt_labels"] = gt_classes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
@PIPELINES.register_module()
class Seg2LabelMaskPanoptic(object):
    def __init__(self,ignore_label=255):
        self.ignore_label = ignore_label
    def __call__(self, results):

        import cv2

        def raster2poly(label_img):
            """Convert binary mask to shapely polygon

            Args:
                label_img (ndarry): bianry mask image np array (N x H x 1)that need to be converted
                contours_img_fn (str): filename of the output conoutrs image
                fg_value (int): the value that represents foreground pixel
                # debug (bool, optional): [description]. Defaults to False.

            Returns:
                polygons , conoturs: shapely polygons and contours of the instance level mask in label_image
            """
            if len(label_img.shape) == 2:
                label_img = label_img[:, :, None]
            label_bi = label_img.astype(np.uint8)
            contours, hierarchy_list = cv2.findContours(
                label_bi, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
            hierarchy = hierarchy_list[0] if hierarchy_list is not None else []
            n_contours = len(contours)
            contours_used_flag = [False] * n_contours
            polygons = []
            ext_contours = []
            int_contours = []
            for i in range(n_contours):
                if contours_used_flag[i]:
                    continue
                contours_used_flag[i] = True
                i_contour = contours[i]
                exterior = i_contour[:, 0, :]
                ext_contours.append(i_contour)
                First_child_index = hierarchy[i][2]
                if len(exterior) <= 2:
                    continue
                polygons.append(exterior)

            return polygons
        seg = results['gt_semantic_seg'].copy()
        h,w = seg.shape[-2:]
        mask = seg > 0
        masks = []
        ps = raster2poly(mask)
        classes=[]
        classes.append(0)
        masks.append((seg==0).astype(np.uint8))
        for i in range(len(ps)):
            temp = np.zeros([h, w], np.uint8)
            temp = cv2.drawContours(temp, [ps[i]], -1, 1, cv2.FILLED)
            masks.append(temp)
            classes.append(1)
        # raise Exception(seg.size())


        # remove ignored region

        gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)

        from mmseg.models.decode_heads.mask2former_utils.structures import BitmapMasks
        results["gt_masks"] = BitmapMasks(masks,h,w)
        results["gt_labels"] = gt_classes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class FlowAddChangeArea(object):
    def __init__(self, keys=['']):

        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            temp = results[key].copy()
            mask = results['gt_semantic_seg'] == 1
            temp[mask]=6
            results[key] = temp


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_key={self.src_key}, '
                     f'dst_keys={self.dst_keys}, '
                     f'dst_values={self.dst_values}, '
                     f'dst_weightmap_values={self.dst_weightmap_values}, '
                     f'ignore_index={self.ignore_index})')
        return repr_str
@PIPELINES.register_module()
class GenPatchLabel(object):
    """Generate patch level label from binary mask.

    Args:
        grid (int, optional): The size of grid. Default: 6
        n_pix_thr (int, optional): The threshold of pixel number. Default: 100
    """

    def __init__(self, grid=6, n_pix_thr=100, ignore_index=255):
        self.grid = grid
        self.n_pix_thr = n_pix_thr
        self.ignore_index = ignore_index

    def _gen_patch_label(self, label, grid):
        patch_label = np.ones((grid, grid)) * 255

        height, width = label.shape
        step_h, step_w = height // grid, width // grid

        for i in range(grid):
            for j in range(grid):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h,
                                   height), min(start_y + step_w, width)
                if i == (grid - 1):
                    end_x = height
                if j == (grid - 1):
                    end_y = width

            temp_label = label[start_x:end_x, start_y:end_y]
            if np.count_nonzero(temp_label == 1) > self.n_pix_thr:
                patch_label[i][j] = 1
            elif np.count_nonzero(temp_label == 1) == 0:
                patch_label[i][j] = 0

        return patch_label.astype(np.int64)

    def __call__(self, results):

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        assert(unique_set.issubset(set([0, 1, self.ignore_index])),
               'Unique of mask is {unique_set}.\n' + \
               'The mask should be binary for GenPatchLabel.')

        results[f'gt_patch{self.grid}_seg'] = self._gen_patch_label(gt_seg, self.grid)
        results['seg_fields'].append(f'gt_patch{self.grid}_seg')

        return results

    def __repr__(self):
        return self.__class__.__name__ \
               + f'(grid={self.grid})' \
               + f'(n_pix_thr={self.n_pix_thr})'

@PIPELINES.register_module()
class GenAreaWeightmap(object):
    """Generate weight map based on instance area.

    Args:
        area_thr (float): The threshold of area.
        wmap_factor (float): The factor of weight map
        generate_key (None or list): The groundtruth to generate Patch Label
    """

    def __init__(self, area_thr, wmap_factor, generate_key=None):
        self.area_thr = area_thr
        self.wmap_factor = wmap_factor
        self.generate_key = generate_key

    def __call__(self, results):
        if 'weightmaps' not in results:
            results['weightmaps'] = {}
        if self.generate_key is not None:
            assert isinstance(self.generate_key, list), "the key must be a list"
            keys = []
            for key in self.generate_key:
                assert key in results['seg_fields'], "the key must be in the seg_fields"
                keys.append(key)
        else:
            keys = results.get('seg_fields', [])

        for key in keys:
            label = results[key]
            unique_set = set(np.unique(label))
            assert unique_set.issubset(set([0, 1, 255]))

            bw = (label == 1)
            bw = clear_border(bw)
            bw_woSmall = remove_small_objects(bw, min_size=self.area_thr)
            bw_Small = np.logical_xor(bw, bw_woSmall)

            factor_map = np.ones_like(label).astype(np.float)
            factor_map[bw_Small == 1] = self.wmap_factor
            factor_map[label == 255] = 0

            if key in results['weightmaps'].keys():
                results['weightmaps'][key] *= factor_map
            else:
                results['weightmaps'][key] = factor_map

        return results

    def __repr__(self):
        return self.__class__.__name__ \
               + f'(area_thr={self.area_thr})' \
               + f'(wmap_factor={self.wmap_factor})'


@PIPELINES.register_module()
class GenWidthWeightmap(object):
    """Generate weight map based on strip (such as road) width.

    Args:
        half_width_thr (float): The threshold of half of width.
        wmap_factor (float): The factor of weight map
        src_key (str, optional): The key of the gt map for generating weightmap.
            Default: 'gt_semantic_seg'.
        dst_keys (list[str], optional): The keys of gt maps for adding weightmap.
            Default: ['gt_semantic_seg'].
    """

    def __init__(self, half_width_thr, wmap_factor, src_key='gt_semantic_seg',
        dst_keys=['gt_semantic_seg']):
        self.half_width_thr = half_width_thr
        self.wmap_factor = wmap_factor
        self.src_key = src_key
        self.dst_keys = dst_keys
        assert isinstance(self.dst_keys, list), "the dst_keys must be a list"

    def __call__(self, results):
        if 'weightmaps' not in results:
            results['weightmaps'] = {}
        assert self.src_key in results['seg_fields'], ("the src_key must be in the seg_fields",
            f'src_key = {src_key}'
            f'dst_keys = {dst_keys}')

        label = results[self.src_key]
        unique_set = set(np.unique(label))
        assert unique_set.issubset(set([0, 1, 255]))

        bw = (label == 1)
        bw_woThin = binary_opening(bw, disk(self.half_width_thr, np.bool))
        bw_Thin = np.logical_xor(bw, bw_woThin)
        bw_Thin = remove_small_objects(bw_Thin, min_size=25)

        factor_map = np.ones_like(label).astype(np.float)
        factor_map[bw_Thin == 1] = self.wmap_factor
        factor_map[label == 255] = 0

        for key in self.dst_keys: 
            if key in results['weightmaps'].keys():
                results['weightmaps'][key] *= factor_map
            else:
                results['weightmaps'][key] = factor_map

        return results

    def __repr__(self):
        return self.__class__.__name__ \
               + f'(half_width_thr={self.half_width_thr})' \
               + f'(wmap_factor={self.wmap_factor})' \
               + f'(src_key={self.src_key})' \
               + f'(dst_keys={self.dst_keys})'

@PIPELINES.register_module()
class BinaryMask2Edge(object):
    """Generate edge map from binary mask

    Args:
        connectivity (int): A pixel is considered a boundary pixel if any of 
            its neighbors has a different label. connectivity controls which
            pixels are considered neighbors. A connectivity of 1 means pixels
            sharing an edge (in 2D) will be considered neighbors.
            A connectivity of 2 means pixels sharing a corner will be
            considered neighbors.
            default is 1.
        mode (str): How to mark the boundaries:
            thick: any pixel not completely surrounded by pixels of the same
                label (defined by connectivity) is marked as a boundary. This
                results in boundaries that are 2 pixels thick.
            inner: outline the pixels just inside of objects, leaving background
                pixels untouched.
            outer: outline pixels in the background around object boundaries.
                When two objects touch, their boundary is also marked.
            default is 'thick'
        ignore_dilation_r (in): Set the dilated ignore area as ignore in edge
            map. The ignore_dilation_r is radius of the dilation operation.
    """

    def __init__(self, connectivity=1, mode='thick', ignore_dilation_r=2,
                 ignore_index=255):
        assert connectivity in {1, 2}
        assert mode in {'thick', 'inner', 'outer'}
        self.connectivity = connectivity
        self.mode = mode
        self.ignore_index = ignore_index
        self.ignore_dilation_r = ignore_dilation_r

    def __call__(self, results):
        """Call function to generate gt edge mask.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Generated gt edge mask, 'gt_edge_seg' key are added into
                result dict.
        """

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        assert(unique_set.issubset(set([0, 1, self.ignore_index])),
               'Unique of mask is {unique_set}.\n' + \
               'The mask should be binary for BinaryMask2Edge.')

        fg = (gt_seg == 1)
        ig = (gt_seg == self.ignore_index)

        edge = find_boundaries(fg, connectivity=self.connectivity,
                               mode=self.mode)

        ig_dilation = binary_dilation(ig, disk(self.ignore_dilation_r, np.bool)) 
        
        edge = edge.astype(np.uint8)
        edge[ig_dilation == 1] = self.ignore_index 

        results['gt_edge_seg'] = edge
        results['seg_fields'].append('gt_edge_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(connectivity={self.connectivity}, mode={self.mode}'
        repr_str += f'(ignore_dilation_r={self.ignore_dilation_r}'
        return repr_str

@PIPELINES.register_module()
class BinaryMask2OrientEdge(object):
    """Generate orient edge map from binary mask

    Args:
        connectivity (int): A pixel is considered a boundary pixel if any of 
            its neighbors has a different label. connectivity controls which
            pixels are considered neighbors. A connectivity of 1 means pixels
            sharing an edge (in 2D) will be considered neighbors.
            A connectivity of 2 means pixels sharing a corner will be
            considered neighbors.
            default is 1.
        mode (str): How to mark the boundaries:
            thick: any pixel not completely surrounded by pixels of the same
                label (defined by connectivity) is marked as a boundary. This
                results in boundaries that are 2 pixels thick.
            inner: outline the pixels just inside of objects, leaving background
                pixels untouched.
            outer: outline pixels in the background around object boundaries.
                When two objects touch, their boundary is also marked.
            default is 'thick'
        num_bins (int | list[int]): The number of bins when dividing the
            [0, 360) angle range.
        blur_sigma (int): Standard deviation for Gaussian kernel when blurring 
            the binary mask. The orientation is obtained by calculating the 
            the gradient of the blurred binary mask.
        ignore_dilation_r (in): Set the dilated ignore area as ignore in edge
            map. The ignore_dilation_r is radius of the dilation operation.
    """

    def __init__(self, connectivity=1, mode='inner', num_bins=[36],
                 blur_sigma=2, ignore_dilation_r=2, ignore_index=255):
        assert connectivity in {1, 2}
        assert mode in {'thick', 'inner', 'outer'}
        self.connectivity = connectivity
        self.mode = mode
        if isinstance(num_bins, list):
            self.num_bins = num_bins
        else:
            self.num_bins = [num_bins]
        assert(set(self.num_bins).issubset(set((4, 8, 18, 36, 45, 90, 180))),
            f'num_bins = {self.num_bins}')
        self.orient_utils = [OrientationUtil(n) for n in self.num_bins]
        self.blur_sigma = blur_sigma

        self.ignore_index = ignore_index
        self.ignore_dilation_r = ignore_dilation_r

    def __call__(self, results):
        """Call function to generated edge.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Generated gt orient edge mask, 'gt_orient_edge_{num_bin}_seg'
                key are added into result dict.
        """

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        assert(unique_set.issubset(set([0, 1, self.ignore_index])),
               'Unique of mask is {unique_set}.\n' + \
               'The mask should be binary for BinaryMask2OrientEdge.')

        fg = (gt_seg == 1)

        ig = (gt_seg == self.ignore_index)
        ig_dilation = binary_dilation(ig, disk(self.ignore_dilation_r, np.bool)) 

        edge = find_boundaries(fg, connectivity=self.connectivity,
                               mode=self.mode)
        edge = edge.astype(np.uint8)
        edge[ig_dilation == 1] = self.ignore_index 

        fg255 = (fg*255).astype(np.uint8)
        fg255_blur = gaussian(fg255, sigma=self.blur_sigma)

        grad_h = sobel_h(fg255_blur)
        grad_v = sobel_v(fg255_blur)

        orient_angle = np.arctan2(grad_v, grad_h) + np.pi
        orient_angle[orient_angle == 2*np.pi] -= 2*np.pi
        assert(np.amax(orient_angle) < 2*np.pi)
        assert(np.amin(orient_angle) >= 0)

        for num_bin, orient_util in zip(self.num_bins, self.orient_utils):
            orient_edge_seg = orient_util.angle_to_label(orient_angle)
            orient_edge_seg = orient_edge_seg.astype(np.uint8)
            orient_edge_seg[edge == 0] = 0
            orient_edge_seg[ig_dilation == 1] = self.ignore_index
            results[f'gt_orient_edge_{num_bin}_seg'] = orient_edge_seg
            results['seg_fields'].append(f'gt_orient_edge_{num_bin}_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(connectivity={self.connectivity}, mode={self.mode}'
        repr_str += f'(num_bins={self.num_bins}'
        repr_str += f'(ignore_dilation_r={self.ignore_dilation_r}'
        return repr_str

@PIPELINES.register_module()
class BinaryMask2EdgeDistance(object):
    """Generate distance map

    Args:
        clip_min (float, optional): Minimum value when clipping. Default: -np.inf 
        clip_max (float, optional): Maximum value when clipping. Default:  np.inf  
        reg_to_cls (bool, optional): Whether to convert regression to classification.
            Default: True.
        num_bins (int, optional): Number of bins when convert regression to
            classification. 
        reg_to_cls_method (str, optional): Method of converting reg to cls.
    """

    def __init__(self,
                 clip_min=-np.inf,
                 clip_max=np.inf,
                 reg_to_cls=True,
                 num_bins=None,
                 reg_to_cls_method='linear',
                 ignore_index=255):
        assert clip_min < clip_max
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reg_to_cls = reg_to_cls
        if self.reg_to_cls:
            assert num_bins > 2
            self.num_bins = num_bins
            self.reg_to_cls_method = reg_to_cls_method
            if self.reg_to_cls_method == 'linear':
                self.bin_width = (self.clip_max - self.clip_min)/(self.num_bins-2) 
            else:
                raise NotImplementedError
        self.ignore_index = ignore_index

    def __call__(self, results):
        """Call function to generate gt edge mask.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Generated gt distance from edge, 'gt_edge_distance_seg' or 
                'gt_edge_distance_reg' key is added into result dict.
        """

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        assert(unique_set.issubset(set([0, 1])),
               'Unique of mask is {unique_set}.\n' + \
               'The mask should be binary for BinaryMask2EdgeDistance.\n' +\
               'Currently, ignored pixels are not supported')
        """
        TODO: If there is any ignored pixel in gt_seg, the whole crop should 
              be ignored for edge_distance.
              So, each map in reg_fields should have the correspoinding weightmap. 
#       assert(unique_set.issubset(set([0, 1, self.ignore_index])),
#              'Unique of mask is {unique_set}.\n' + \
#              'The mask should be binary for BinaryMask2EdgeDistance.')
        """

        fg = (gt_seg == 1)
        if np.count_nonzero(np.logical_not(fg)) == 0:
            H, W = fg.shape[:2]
            edge_distance = np.ones_like(fg) * np.sqrt(H*H + W*W)
        else:
            edge_distance = ndi.distance_transform_edt(fg)

        if self.reg_to_cls:
            if self.reg_to_cls_method == 'linear':
                edge_distance_seg = (edge_distance - self.clip_min) // self.bin_width + 1
                edge_distance_seg[edge_distance <= self.clip_min] = 0
                edge_distance_seg[edge_distance >= self.clip_max] = self.num_bins-1
            else:
                raise NotImplementedError
            results['gt_edge_distance_seg'] = edge_distance_seg
            results['seg_fields'].append('gt_edge_distance_seg')
        else:
            edge_distance = np.clip(edge_distance, self.clip_min, self.clip_max)
            edge_distance = np.expand_dims(edge_distance, 0)
            results['gt_edge_distance_reg'] = edge_distance
            results['reg_fields'].append('gt_edge_distance_reg')
        return results


@PIPELINES.register_module()
class Mask2EdgeDistance(object):
    """Generate distance map

    Args:
        ch_ind (list[int]): index of channels for generating edge distance.
        clip_min (float, optional): Minimum value when clipping. Default: -np.inf 
        clip_max (float, optional): Maximum value when clipping. Default:  np.inf  
    """

    def __init__(self,
                 ch_ind=[1],
                 clip_min=-np.inf,
                 clip_max=np.inf,
                 ignore_index=255):
        assert isinstance(ch_ind, list) 
        self.ch_ind = ch_ind
        assert clip_min < clip_max
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.ignore_index = ignore_index

    def __call__(self, results):
        """Call function to generate gt edge mask.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Generated gt distance from edge, 'gt_edge_distance_seg' or 
                'gt_edge_distance_reg' key is added into result dict.
        """

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        assert(unique_set.issubset(set(range(max(self.ch_ind)+1))),
               'Unique of mask is {unique_set}.\n'
               'self.ch_ind = {self.ch_ind}.\n' 
               'Currently, ignored pixels are not supported')

        """
        TODO: If there is any ignored pixel in gt_seg, the whole crop should 
              be ignored for edge_distance.
              So, each map in reg_fields should have the correspoinding weightmap. 
        """

        H, W = gt_seg.shape[:2]
        edge_distance = np.zeros((len(self.ch_ind), H, W))

        for i in self.ch_ind:
            fg = (gt_seg == i)
            if np.count_nonzero(np.logical_not(fg)) == 0:
                ed = np.ones_like(fg) * np.sqrt(H*H + W*W)
            else:
                ed = ndi.distance_transform_edt(fg)
            edge_distance[i, ...] = np.clip(ed, self.clip_min, self.clip_max)

        results['gt_edge_distance_reg'] = edge_distance
        results['reg_fields'].append('gt_edge_distance_reg')
        return results

@PIPELINES.register_module()
class BinaryMask2OrientRoad(object):
    """Generate orient road map from orient edge

    Args:
        connectivity (int): A pixel is considered a boundary pixel if any of 
            its neighbors has a different label. connectivity controls which
            pixels are considered neighbors. A connectivity of 1 means pixels
            sharing an edge (in 2D) will be considered neighbors.
            A connectivity of 2 means pixels sharing a corner will be
            considered neighbors.
            default is 1.
        mode (str): How to mark the boundaries:
            thick: any pixel not completely surrounded by pixels of the same
                label (defined by connectivity) is marked as a boundary. This
                results in boundaries that are 2 pixels thick.
            inner: outline the pixels just inside of objects, leaving background
                pixels untouched.
            outer: outline pixels in the background around object boundaries.
                When two objects touch, their boundary is also marked.
            default is 'thick'
        num_bins (int | list[int]): The number of bins when dividing the
            [0, 360) angle range.
        blur_sigma (int): Standard deviation for Gaussian kernel when blurring 
            the binary mask. The orientation is obtained by calculating the 
            the gradient of the blurred binary mask.
        ignore_dilation_r (in): Set the dilated ignore area as ignore in edge
            map. The ignore_dilation_r is radius of the dilation operation.
    """

    def __init__(self, connectivity=1, mode='inner', num_bins=[36],
                 blur_sigma=2, ignore_dilation_r=2, ignore_index=255):
        assert connectivity in {1, 2}
        assert mode in {'thick', 'inner', 'outer'}
        self.connectivity = connectivity
        self.mode = mode
        if isinstance(num_bins, list):
            self.num_bins = num_bins
        else:
            self.num_bins = [num_bins]
        assert(set(self.num_bins).issubset(set((4, 8, 18, 36, 45, 90, 180))),
            f'num_bins = {self.num_bins}')
        self.orient_utils = [OrientationUtil(n,np.pi) for n in self.num_bins]
        self.blur_sigma = blur_sigma

        self.ignore_index = ignore_index
        self.ignore_dilation_r = ignore_dilation_r

    def __call__(self, results):
        """Call function to generate orient road.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Generated gt orient road mask, 'gt_orient_road_{num_bin}_seg'
                key are added into result dict.
        """

        assert 'gt_semantic_seg' in results
        gt_seg = results['gt_semantic_seg']

        unique_set = set(np.unique(gt_seg))
        # assert(unique_set.issubset(set([0, 1, self.ignore_index])),
        #        'Unique of mask is {unique_set}.\n' + \
        #        'The mask should be binary for BinaryMask2OrientEdge.')

        fg = (gt_seg == 1)

        ig = (gt_seg == self.ignore_index)
        ig_dilation = binary_dilation(ig, disk(self.ignore_dilation_r, np.bool)) 

        edge = find_boundaries(fg, connectivity=self.connectivity,
                               mode=self.mode)
        edge = edge.astype(np.uint8)
        edge[ig_dilation == 1] = self.ignore_index 

        fg255 = (fg*255).astype(np.uint8)
        fg255_blur = gaussian(fg255, sigma=self.blur_sigma)

        grad_h = sobel_h(fg255_blur)
        grad_v = sobel_v(fg255_blur)

        orient_angle = np.arctan2(grad_v, grad_h) + np.pi
        orient_angle[orient_angle == 2*np.pi] -= 2*np.pi
        assert(np.amax(orient_angle) < 2*np.pi)
        assert(np.amin(orient_angle) >= 0)

        orient_angle[orient_angle >= np.pi] -= np.pi
        for num_bin, orient_util in zip(self.num_bins, self.orient_utils):
            orient_edge_seg = orient_util.angle_to_label(orient_angle)
            orient_edge_seg = orient_edge_seg.astype(np.uint8)
            orient_edge_seg[edge == 0] = 0
            orient_edge_seg[ig_dilation == 1] = self.ignore_index
        
        fg = fg.astype(np.uint8)

        # create knn train set based on orient_edge_seg
        edge_index = np.argwhere(orient_edge_seg > 0)
        edge_label = orient_edge_seg[orient_edge_seg > 0]

        if edge_label.shape[0] > 1:
            knn = neighbors.KNeighborsClassifier(n_neighbors=1)
            knn.fit(edge_index,edge_label)
            fg_index = np.argwhere(fg > 0)
            if fg_index.shape[0] > 1:
                fg[fg > 0] = knn.predict(fg_index)

        orient_road_seg = fg.copy()
        orient_road_seg[fg == 0] = 0
        orient_road_seg[ig_dilation ==1] = self.ignore_index
        results[f'gt_orient_road_{num_bin}_seg'] = orient_road_seg
        results['seg_fields'].append(f'gt_orient_road_{num_bin}_seg')
 
        return results



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(connectivity={self.connectivity}, mode={self.mode}'
        repr_str += f'(num_bins={self.num_bins}'
        repr_str += f'(ignore_dilation_r={self.ignore_dilation_r}'
        return repr_str
