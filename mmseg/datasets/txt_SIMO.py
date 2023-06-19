import os.path as osp
from pathlib import Path
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class TxtSIMODataset(Dataset):
    """Semantic segmentation dataset for single input (image)
       and multiple output (gt map).

    The dataset is specified by the text file.

    Args:
        pipeline (list[dict]): Processing pipeline
        txt_fn (str): Filename of text file.
        data_root (str|None, optional): Data root for paths in txt_fn. Default: None
        test_mode (str): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (list[str]|None): Names of classes. Default: None
        palette (list[list[int]] | None): Palettes of classes for output.
            Default: None
    """

    def __init__(self,
                 pipeline,
                 txt_fn,
                 anno_name_list=['seg_map'],
                 data_root="",
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.txt_fn = txt_fn
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        assert osp.isfile(txt_fn), f'{txt_fn} NOT exists!'
        self.anno_name_list=anno_name_list
        # load annotations
        self.img_infos = self.load_annotations(self.data_root, self.txt_fn)

        self.CLASSES = classes
        self.PALETTE = palette
        if self.PALETTE is not None:
            self.PALETTE = np.array(self.PALETTE)
            assert self.PALETTE.shape[1] == 3
            if self.CLASSES is not None:
                assert isinstance(self.CLASSES, list)
                assert len(self.CLASSES) == self.PALETTE.shape[0]

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, data_root, txt_fn):
        """Load annotation from directory.
        SISO:'filename':imgName,fn_list[0],'ann':seg_map,fn_list[1]
        MISO:'filename':[imgName1,imgName2],line_split[:-1];'ann':seg_map,line_split[-1]
        SIMO:'filename':imgName,fn_list[0],'ann':[seg_map,reg_map,xxxx]
        MI:list  MO:dict

        Args:
            data_root (str): Data root for paths in txt_fn.
            txt_fn (str): Filename of text file.

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        with open(txt_fn) as f:
            for line in f:
                fn_lst = line.strip().split(' ')
                img_file = osp.join(data_root, fn_lst[0])
                img_info = dict(filename=img_file)
                img_info['ann']={}
                if len(fn_lst) >= 2:
                    assert len(self.anno_name_list)==len(fn_lst[1:]),'%s %s; %s %s'%('anno_name_list is',anno_name_list,'fn_lst is',fn_lst)
                    seg_map = osp.join(data_root, fn_lst[1])
                    for anno_name,anno_path in zip(anno_name_list,fn_lst[1:]):
                        img_info['ann'][anno_name] = anno_path
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['reg_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            gt_seg_map = mmcv.imread(
                img_info['ann']['seg_map'], flag='unchanged', backend='pillow')
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        all_acc, acc, iou = mean_iou(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format('global', iou_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results

    def get_pred_seg_maps(self, pred_dir):
        pred_seg_maps = []
        for img_info in self.img_infos:
            gt_fn = img_info['ann']['seg_map']
            if isinstance(gt_fn, Path):
                gt_fn = str(gt_fn)
            pred_fn = osp.join(pred_dir, osp.basename(gt_fn))
            if not osp.isfile(pred_fn):
                raise FileNotFoundError("{} not Found!".format(pred_fn))
            else:
                pred_seg_map = mmcv.imread(
                    pred_fn, flag='unchanged', backend='pillow')
                pred_seg_maps.append(pred_seg_map)

        return pred_seg_maps

    def get_gt_dirs(self):
        gt_dirs = []
        for img_info in self.img_infos:
            gt_fn = img_info['ann']['seg_map']
            if isinstance(gt_fn, Path):
                gt_fn = str(gn_fn)
            gt_dir = osp.dirname(gt_fn)
            gt_dirs.append(gt_dir)
        gt_dirs = list(set(gt_dirs))
        return gt_dirs

    def evaluate_dist(self, pred_dir, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            pred_dir (str): Directory which stores the predicted result.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'pixwise_foreground']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        if metric == 'mIoU':
            results = self.get_pred_seg_maps(pred_dir)
            return self.evaluate(results, metric, logger, **kwargs)
        elif metric == 'pixwise_foreground':

            # !!!!!!!!!!!!! To be optimized in the future !!!!!!!!!!!!!!!
            import sys
            sys.path.insert(0, '/mnt/lustre/wujiang/building/instance_evaluation')
            import sub_1_post_process
            import sub_2_eval_pix

            pp_dir = pred_dir
#           # --------------------------------- post-process
#           param  = {}
#
#           param['method']      = 'removing_small_hole_and_object'
#           param['hole_thr']   = 9     # step1: removing small hole
#           param['object_thr'] = 100   # step2: removing small instance
#           pp_dir = '{}_rmHole{}Obj{}'.format(pred_dir, param['hole_thr'], param['object_thr'])
#           sub_1_post_process.post_process(pred_dir, None, pp_dir, param)

            # --------------------------------- eval
            gt_dirs = self.get_gt_dirs()
            if len(gt_dirs) != 1:
                print("The gt files should be stored in ONE folder!")
                print("gt_dirs = {}".format(gt_dirs))
                raise RuntimeError
            else:
                gt_dir = gt_dirs[0]

            eval_out_root = pp_dir + '_evalPix'
            sub_2_eval_pix.evaluate(gt_dir,  pp_dir, eval_out_root, prd_format='0_1')
            sub_2_eval_pix.summarize(gt_dir, pp_dir, eval_out_root)

        elif metric == 'inswise_foreground':

            # !!!!!!!!!!!!! To be optimized in the future !!!!!!!!!!!!!!!
            import sys
            sys.path.insert(0, '/mnt/lustre/wujiang/dituo/instance_evaluation__dev')
            import sub_1_post_process
            import sub_2_eval

            # --------------------------------- post-process
            param_eval = {}

            param_eval['prd_format']  = '0_1'
            param_eval['method']      = 'closing_and_removing_small'
            param_eval['imclose_r']   = 10     # step1: closing (erode after dilating) the initial mask
            param_eval['im_areaopen'] = 100    # step2: removing small instance
            pp_dir = pred_dir + '_imclsR' + str(param_eval['imclose_r']) + 'ao' + str(param_eval['im_areaopen'])

            print("pp_dir = {}".format(pp_dir))
            sub_1_post_process.post_process(pred_dir, "", pp_dir, param_eval)

            # --------------------------------- eval
            gt_dirs = self.get_gt_dirs()
            if len(gt_dirs) != 1:
                print("The gt files should be stored in ONE folder!")
                print("gt_dirs = {}".format(gt_dirs))
                raise RuntimeError
            else:
                gt_dir = gt_dirs[0]

            param_eval['gt_ins_area_min'] = 100    # minimum area of valid instance for gt (the instance of gt whose area is lower than 'gt_ins_area_min' will be removed when evaluating)
            param_eval['ins_IoU_thr'] = 0.1

            gt_ins_area_list = list(set([0, 100, 200, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, param_eval['gt_ins_area_min']]))
            gt_ins_area_list.sort()
            param_eval['gt_ins_area_list'] = gt_ins_area_list   # list of thresholds to compute recalls

            eval_out_root = pp_dir + '_GtInsMin' + str(param_eval['gt_ins_area_min']) +  '_insIoUthr' + str(param_eval['ins_IoU_thr']) + '_nonStrict'

#           sub_2_eval.evaluate( gt_dir, pp_dir, eval_out_root, param_eval, isStrict=False, crop_FP=crop_FP, img1_dir=img1_dir, img2_dir=img2_dir, img_suf=img_suf)
            sub_2_eval.evaluate( gt_dir, pp_dir, eval_out_root, param_eval, isStrict=False)
            sub_2_eval.summarize(gt_dir, pp_dir, eval_out_root)

        else:
            raise NotImplementedError
