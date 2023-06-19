import os, sys
import os.path as osp

import mmcv
import numpy as np
import torch

from mmseg.datasets.pipelines import OrientationUtil

def change_test_prob_save(model, data_loader, out_root, output_prob=False,
                     output_shared_extra_head=False):
    """Test with single GPU (batchsize = 1) and save result in two 
       directories "gray" and "cmap".

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        out_root (str): The root directory for saving results.
        output_prob (bool): Whether to output the probability. Default: False.
        output_shared_extra_head (bool): Whether to output shared extra head.

    Returns:
        list: The output diectory.
    """

    model.eval()
    dataset = data_loader.dataset

    out_dir0 = osp.join(out_root, "prob")
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

    if output_shared_extra_head:
        out_dir_extra1 = osp.join(out_root, 'shared_extra_1')
        out_dir_extra2 = osp.join(out_root, 'shared_extra_2')
        mmcv.mkdir_or_exist(out_dir_extra1)
        mmcv.mkdir_or_exist(out_dir_extra2)

        extra_palette = None
        if model.module.EXTRA_PALETTE is not None:
            extra_palette = model.module.EXTRA_PALETTE 
            out_dir_extra1_color = osp.join(out_root, 'shared_extra_1_cmp')
            out_dir_extra2_color = osp.join(out_root, 'shared_extra_2_cmp')
            mmcv.mkdir_or_exist(out_dir_extra1_color)
            mmcv.mkdir_or_exist(out_dir_extra2_color)
   
    prog_bar = mmcv.ProgressBar(len(dataset))
    for index,data in enumerate(data_loader):
        batch_size = data['img'][0].size(0)
        assert batch_size == 1

        img_meta = data['img_metas'][0].data[0][0]
        # TODO: optimize ori_filename
        if isinstance(img_meta['ori_filename'], list):
            ori_filename = img_meta['ori_filename'][0]
        else:
            ori_filename = img_meta['ori_filename']
        fn_pre = osp.splitext(osp.basename(ori_filename))[0]
        # out_fn1 = osp.join(out_dir0, fn_pre+'_'+str(index)+'.npy')
        out_fn1 = osp.join(out_dir0, fn_pre + '.npy')
        print(index,fn_pre)


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
            result = model(return_loss=False, rescale=True, **data)[0]

        if output_shared_extra_head:
            assert len(result) == 3
        else:
            print(result[0].shape)

        pred = result[0]

        np.save(out_fn1,pred)

        if osp.isfile(tmp_fn):
            try:
                os.remove(tmp_fn)
            except:
                pass

        prog_bar.update()

    return out_dir0

