import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multihead_test_save, change_test_mtl_index_save
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('test_data_config', help='test data config file path')
#   parser.add_argument(
#       '--aug_test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--out_dir', help='directory where inference results will be saved')
    parser.add_argument(
        '--output-prob', action='store_true', help='output the probability')
    parser.add_argument(
        '--output-shared-extra-head', action='store_true',
        help='output the shared extra head')
    parser.add_argument(
        '--eval_options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    assert args.out_dir, \
        ('Please specify the folder to output with the argument "--out-dir"')

    cfg = mmcv.Config.fromfile(args.config)
    test_data_cfg = mmcv.Config.fromfile(args.test_data_config)

    # copy crop_size and stride for 'slide' test mode from cfg
    if cfg.test_cfg.mode == 'slide':
        if 'crop_size' not in cfg.test_cfg:
            cfg.test_cfg.crop_size = cfg.crop_size
        if 'stride' not in cfg.test_cfg:
            cfg.test_cfg.stride    = list(int(_*cfg.test_cfg.stride_ratio)
                for _ in cfg.crop_size)

    if args.output_shared_extra_head:
        cfg.test_cfg.output_shared_extra_head = True

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
#   if args.aug_test:
#       # hard code index
#       cfg.data.test.pipeline[1].img_ratios = [
#           0.5, 0.75, 1.0, 1.25, 1.5, 1.75
#       ]
#       cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    test_data_cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(test_data_cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    output_shared_extra_head = False
    if hasattr(cfg.test_cfg, 'output_shared_extra_head') and \
        cfg.test_cfg.output_shared_extra_head:
            output_shared_extra_head = True
            if hasattr(cfg.data, 'extra_classes') and hasattr(cfg.data, 'extra_palette'):
                model.EXTRA_CLASSES = cfg.data.extra_classes
                model.EXTRA_PALETTE = cfg.data.extra_palette
            else:
                model.EXTRA_CLASSES = None
                model.EXTRA_PALETTE = None

    model = MMDataParallel(model, device_ids=[0])
    mmcv.mkdir_or_exist(args.out_dir)
    cfg.dump(osp.join(args.out_dir, 'train_config.py'))
    test_data_cfg.dump(osp.join(args.out_dir, 'test_data_config.py'))

    if test_data_cfg.data.test.type in {'TxtMISODataset'}:
        out_dir_gray = change_test_mtl_index_save(model, data_loader, args.out_dir,
            args.output_prob, output_shared_extra_head)
    else:
        # out_dir_gray = multihead_test_save(model, data_loader, args.out_dir,
        #     args.output_prob)
        out_dir_gray = change_test_mtl_index_save(model, data_loader, args.out_dir,
                                            args.output_prob, output_shared_extra_head)

    out_dir_gray = osp.join(args.out_dir, "gray")
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.eval:
        dataset.evaluate_dist(out_dir_gray, args.eval, **kwargs)


if __name__ == '__main__':
    main()
