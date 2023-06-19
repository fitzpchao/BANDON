import os, sys
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmseg.datasets.pipelines import OrientationUtil

def ds_cd_save(model, data_loader, out_root, output_prob=False,
                     output_shared_extra_head=False,mean=[0,0,0],std=[255,255,255]):
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


    out_dir1 = osp.join(out_root, "gray")
    mmcv.mkdir_or_exist(out_dir1)

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




   
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        batch_size = data['img'][0].size(0)
        assert batch_size == 1

        img_meta = data['img_metas'][0].data[0][0]
        # TODO: optimize ori_filename
        if isinstance(img_meta['ori_filename'], list):
            ori_filename = img_meta['ori_filename'][0]
        else:
            ori_filename = img_meta['ori_filename']
        fn_pre = osp.splitext(osp.basename(ori_filename))[0]
        out_fn1 = osp.join(out_dir1, fn_pre+'.png')
        if palette is not None:
            out_fn2 = osp.join(out_dir0_color, fn_pre + '.png')

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
            result = model.slide_inference_mtl(indexs_list=[0],resize_list=[True],num_classes_list=[2], **data)
        result = result[0][0]
        output = F.softmax(result, dim=0)
        output = output.cpu().detach().numpy()
        seg = np.argmax(output, axis=0)

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





        if osp.isfile(tmp_fn):
            try:
                os.remove(tmp_fn)
            except:
                pass

        prog_bar.update()

    return

