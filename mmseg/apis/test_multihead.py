import os, sys
import os.path as osp

import mmcv
import numpy as np
import torch
import rasterio

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from rasterio.plot import reshape_as_raster, reshape_as_image
from affine import Affine

from mmseg.datasets.pipelines import OrientationUtil
from mmseg.datasets.vis import show

def multihead_test_save(model, data_loader, out_root, output_prob=False):
    """Test with single GPU (batchsize = 1) and save result in two
       directories "gray" and "cmap".

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        out_root (str): The root directory for saving results.
        output_prob (bool): Whether to output the probability. Default: False.

    Returns:
        list: The output diectory.
        'gray':save the seg map,0,1,2,3
        'cmap':save gray in palette coloe

    """

    model.eval()
    dataset = data_loader.dataset

    out_dir0 = osp.join(out_root, "gray")
    mmcv.mkdir_or_exist(out_dir0)

    if dataset.PALETTE is not None:
        palette = dataset.PALETTE
    elif model.module.PALETTE is not None:
        palette = model.module.PALETTE
    else:
        palette = None

    if palette is not None:
        palette = np.array(palette)
#       assert palette.shape[0] == len(model.module.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        out_dir0_color = osp.join(out_root, "cmap")
        mmcv.mkdir_or_exist(out_dir0_color)

    if output_prob:
        out_dir0_prob = out_dir0 + '_prob'
        mmcv.mkdir_or_exist(out_dir0_prob)

    num_heads = model.module.num_heads
    if hasattr(model.module, 'postprocess_keys'):
        postprocess_keys = model.module.postprocess_keys
    else:
        postprocess_keys = None
    out_dirs = []

    if num_heads > 1:
        try:
            gt_keys = model.module.gt_keys
        except:
            raise RuntimeError('gt_keys should be defined in segmentor')

        for key in gt_keys[1:]:
            key = key.split('_')
            assert key[0] == 'gt'
            assert key[-1] in {'seg', 'reg'}
            out_dirs.append('_'.join(key[1:-1]))

        try:
            palettes_mh = model.module.get_palettes()
        except:
            palettes_mh = None
        if palettes_mh is not None:
            assert isinstance(palettes_mh, list)
            for i in range(len(palettes_mh)):
                if palettes_mh[i] is not None:
                    palettes_mh[i] = np.array(palettes_mh[i])
                    assert palettes_mh[i].shape[1] == 3
                    assert len(palettes_mh[i].shape) == 2

    if postprocess_keys is not None:
        pp_keys = [key for keys in postprocess_keys for key in keys]
        for key in pp_keys:
            if key.startswith('gt_orient_edge_') and key.endswith('_seg'):
                out_dirs.append('_'.join(key.split('_')[1:-1]))
            else:
                raise NotImplementedError
    else:
        pp_keys = None

    if out_dirs:
        out_dirs = [osp.join(out_root, d) for d in out_dirs]
        out_color_dirs = [d + '_color' for d in out_dirs]
        map(mmcv.mkdir_or_exist, out_dirs)
        map(mmcv.mkdir_or_exist, out_color_dirs)
        if output_prob:
            out_prob_dirs = [d + '_prob' for d in out_dirs]
            map(mmcv.mkdir_or_exist, out_prob_dirs)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        batch_size = data['img'][0].size(0)
        assert batch_size == 1

        img_meta = data['img_metas'][0].data[0][0]
        fn_pre = osp.splitext(osp.basename(img_meta['ori_filename']))[0]
        out_fn1 = osp.join(out_dir0, fn_pre+'.png')
        if palette is not None:
            out_fn2 = osp.join(out_dir0_color, fn_pre+'.png')

        if osp.isfile(out_fn1):
            prog_bar.update()
            continue
        tmp_fn = out_fn1 + ".tmp"
        if osp.isfile(tmp_fn):
            prog_bar.update()
            continue
        else:
            open(tmp_fn,'w').close()

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        out_lst = result[0]
        if not isinstance(out_lst, list):
            out_lst = [out_lst]
        assert len(out_lst) == num_heads + (len(pp_keys) if pp_keys else 0)
        pred = out_lst[0]
        if output_prob:
            for c in range(pred.shape[0]):
                p = pred[c, :, :]
                assert np.amax(p) <= 1.0
                assert np.amin(p) >= 0.0
                p_out = (p*255).astype(np.uint8)
                mmcv.imwrite(p_out, osp.join(out_dir0_prob, str(c), fn_pre+'.png'))

        seg = np.argmax(pred, axis=0)
        if palette is not None:
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            # convert to BGR
            color_seg = color_seg[..., ::-1]

        seg = seg.astype(np.uint8)
        mmcv.imwrite(seg, out_fn1)

        if palette is not None:
            color_seg = color_seg.astype(np.uint8)
            mmcv.imwrite(color_seg, out_fn2)

        # -------------------------------------- output for multiple head
        for i in range(1, num_heads):
            if not model.module.decode_head[i].is_reg:
                out = out_lst[i].astype(np.uint8)
                mmcv.imwrite(out, osp.join(out_dirs[i-1], fn_pre+'.png'))

                if gt_keys[i] == 'gt_edge_seg':
                    color_out = (out*255).astype(np.uint8)
                    mmcv.imwrite(color_out, osp.join(out_color_dirs[i-1], fn_pre+'.png'))
                elif gt_keys[i].startswith('gt_orient_edge_') \
                     and gt_keys[i].endswith('_seg'):
                    num_bin = int(gt_keys[i].split('_')[3])
                    ort_util = OrientationUtil(num_bin)
                    color_out = ort_util.label_to_color(out)
                    color_out = color_out[..., ::-1] # convert to BGR
                    mmcv.imwrite(color_out, osp.join(out_color_dirs[i-1], fn_pre+'.png'))
                elif gt_keys[i].startswith('gt_orient_road_') \
                     and gt_keys[i].endswith('_seg'):
                    num_bin = int(gt_keys[i].split('_')[3])
                    ort_util = OrientationUtil(num_bin, max_angle=np.pi)
                    color_out = ort_util.label_to_color(out)
                    color_out = color_out[..., ::-1] # convert to BGR
                    mmcv.imwrite(color_out, osp.join(out_color_dirs[i-1], fn_pre+'.png'))
                elif (palettes_mh is not None) and (palettes_mh[i] is not None):
                    color_out = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
                    for label, color in enumerate(palettes_mh[i]):
                        color_out[out == label, :] = color
                    color_out = color_out[..., ::-1] # convert to BGR
                    mmcv.imwrite(color_out, osp.join(out_color_dirs[i-1], fn_pre+'.png'))
            else:
                #save the reg offset in float_tif and vis_jpg
                out = out_lst[i]#out.shape is C,H,W
                scale_list=[_.data[0][0]['scale_factor'][0] for _ in data['img_metas']]
                C, H, W = out.shape
                offset_dst_dir =osp.join(out_dirs[i-1], 'label_flow')
                os.makedirs(offset_dst_dir,exist_ok=True)
                assert out.ndim == 3
                if gt_keys[i] == 'gt_label_flow_reg':
                    #save as float32 tif file
                    with rasterio.Env():
                        profile={'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': W, 'height': H, 'count': C, 'crs': None, 'transform': Affine(1.0, 0.0, 0.0,0.0, 1.0, 0.0), 'tiled': False, 'interleave': 'pixel'}
                        profile.update(dtype=rasterio.float32,compress='lzw')
                        with rasterio.open(osp.join(offset_dst_dir, fn_pre+'.tif'),'w',**profile) as dst_dataset:
                            out=out/np.sum(scale_list)
                            dst_dataset.write(out)
                    mmcv.imwrite(show.reg_to_img(out),osp.join(offset_dst_dir, fn_pre+'.jpg'))

                #C, H, W = out.shape
                #font_path = osp.join(osp.dirname(osp.realpath(__file__)), "arial.ttf")
                #font = ImageFont.truetype(font_path, 60 if min(H,W)>1500 else 20)
                #for c in range(C):
                    #c_max = np.amax(out[c, ...])
                    #c_min = np.amin(out[c, ...])
                    #c_out = (out[c, ...] - c_min)/(c_max - c_min)
                    #c_out = (c_out*255).astype(np.uint8)
                    #c_out = Image.fromarray(c_out, 'L')
                    #draw = ImageDraw.Draw(c_out)
                    #draw.text((10, 10), f'min = {c_min}, max = {c_max}', 'white', font=font)
                    #c_dst_dir =osp.join(out_dirs[i-1], str(c))
                    #os.makedirs(c_dst_dir, exist_ok=True)
                    #c_out.save(osp.join(c_dst_dir, fn_pre+'.png'))

        # -------------------------------------- output for post-processed output
        if pp_keys is not None:
            for j in range(len(pp_keys)):
                i = j + num_heads
                out = out_lst[i].astype(np.uint8)
                mmcv.imwrite(out, osp.join(out_dirs[i-1], fn_pre+'.png'))

                pp_key = pp_keys[j]
                if (    pp_key.startswith('gt_orient_edge_')
                    and pp_key.endswith('_seg')
                ):
                    pp_num_bin = int(pp_key.split('_')[3])
                    ort_util = OrientationUtil(pp_num_bin)
                    color_out = ort_util.label_to_color(out)
                    color_out = color_out[..., ::-1] # convert to BGR
                    mmcv.imwrite(color_out, osp.join(out_color_dirs[i-1], fn_pre+'.png'))

        if osp.isfile(tmp_fn):
            try:
                os.remove(tmp_fn)
            except:
                pass

        prog_bar.update()

    return out_dir0

