import sys
import os.path as osp
import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from torch import nn
import torch._C
import torch.serialization
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor

torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 skip_exist=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    img_list = [img[None, :] for img in imgs]
    img_meta_list = [[img_meta] for img_meta in img_metas]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward, img_metas=img_meta_list, return_loss=False)

    register_extra_symbolics(opset_version)
    if skip_exist and osp.isfile(output_file):
        print(f'ONNX file: {output_file} already exists!!!')
    else:
        with torch.no_grad():
            torch.onnx.export(
                model, (img_list, ),
                output_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                verbose=show,
                opset_version=opset_version)
            print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_meta_list, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]
        if not np.allclose(pytorch_result, onnx_result):
            print(f'pytorch_result.shape = {pytorch_result.shape}')
            print(f'onnx_result.shape = {onnx_result.shape}')
            print(f'pytorch_result.size = {pytorch_result.size}')
            print(f'onnx_result.size = {onnx_result.size}')
            print( f'np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-05, atol=1e-08)) = ' \
                + f'{np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-05, atol=1e-08))}')
            print( f'np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-04, atol=1e-07)) = ' \
                + f'{np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-04, atol=1e-07))}')
            print( f'np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-03, atol=1e-05)) = ' \
                + f'{np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-03, atol=1e-05))}')
            print( f'np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-02, atol=1e-04)) = ' \
                + f'{np.count_nonzero(np.isclose(pytorch_result, onnx_result, rtol=1e-02, atol=1e-04))}')

            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSeg to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--skip-exist', action='store_true',
        help='skip the convert when the onnx file exists')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--input-channel', type=int, default=3)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, args.input_channel, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            args.input_channel,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes

    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        segmentor,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        skip_exist=args.skip_exist)
