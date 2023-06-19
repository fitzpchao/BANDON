import os, sys
import os.path as osp

import mmcv
import numpy as np
import torch

from mmseg.datasets.pipelines import OrientationUtil

def sr_save(model, data_loader, out_root, output_prob=False,
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


    out_dir1 = osp.join(out_root, "sr")
    mmcv.mkdir_or_exist(out_dir1)

    if dataset.PALETTE is not None:
        palette = dataset.PALETTE
    elif model.module.PALETTE is not None:
        palette = model.module.PALETTE
    else:
        palette = None
 




   
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
            result = model.slide_inference_mtl(indexs_list=[0],resize_list=[True],num_classes_list=[3], **data)


        pred_sr = result[0][0].cpu().detach().numpy().transpose([1,2,0])
        for c in range(pred_sr.shape[-1]):

            pred_sr[:,:,c] = pred_sr[:,:,c] * std[c] + mean[c]
        pred_sr = pred_sr[..., ::-1].astype(np.uint8)
        mmcv.imwrite(pred_sr, out_fn1)





        if osp.isfile(tmp_fn):
            try:
                os.remove(tmp_fn)
            except:
                pass

        prog_bar.update()

    return

