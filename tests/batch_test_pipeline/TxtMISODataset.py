import os, sys
import os.path as osp
import numpy as np
import cv2
import mmcv
from mmseg.datasets import TxtMISODataset
from mmseg.datasets import build_dataset, build_dataloader
PALETTE = { 
    #     b    g    r
    0  : (0  , 0  , 255),
    1  : (0  , 255, 0  ),
    2  : (255, 0  , 0  ),
    3  : (0  , 255, 255),
    4  : (255, 255, 0  ),
    5  : (255, 0  , 255),
    6  : (0  , 0  , 150),
    7  : (0  , 150, 0  ),
    8  : (150, 0  , 0  ),
    9  : (0  , 150, 150),
    10 : (150, 150, 0  ),
    11 : (150, 0  , 150),
    12 : (0  , 0  , 50 ),
    13 : (0  , 50 , 0  ),
    14 : (50 , 0  , 0  ),
    15 : (0  , 50 , 50 ),
    16 : (50 , 50 , 0  ),
    17 : (50 , 0  , 50 ),
    255: (255, 255, 255)
}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> config
dataset_type = 'TxtMISODataset'
data_root = ''
img_norm_cfg = dict(
    mean = [127, 127, 127],
    std  = [58, 58, 58],
    to_rgb=True
)
#crop_size = (512, 512)
crop_size = (321, 321)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=None, ratio_range=(0.2, 0.4)),
#   dict(type='Resize', img_scale=None, ratio_range=(0.7, 1.5)),
    dict(type='RandomRotate', rotate_range=(-180, 180)),
    dict(type='RandomCrop', crop_size=list(c + 20 for c in crop_size)),
    dict(type='CenterCrop', crop_size=crop_size, rand_pair_trans_offset=10),
    dict(type='RandomGaussianBlur', blur_ratio=0.5, radius=5),
    dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type="RandomRotate90n"),
    dict(type='PhotoMetricDistortion',
        brightness_delta=20,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.7, 1.3)),
    dict(type='RandomPixelNoise', max_pixel_noise=20),
    dict(type='DecodeLabel',
        src_key='gt_semantic_seg',
        dst_keys=['gt_semantic_seg'],
        dst_values=[[0, 0, 1, 1]],
        dst_weightmap_values=[[1, 200, 1, 30]]),
    dict(type="GenPatchLabel"),
    dict(type="GenAreaWeightmap", area_thr=400, wmap_factor=2.),
    dict(type="GenAreaWeightmap", area_thr=900, wmap_factor=2.),

#   dict(type='Normalize', **img_norm_cfg),
#   dict(type='DefaultFormatBundle'),
    dict(type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_patch6_seg', 'weightmaps']
    )
]
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        txt_fn = '/mnt/lustre/wujiang/buildchange/data/list/' \
                 + 'cat__FR01_PHR_ShWhJn_x2__L17v1_chzx_bjDOM_SPOT.txt',
        pipeline=train_pipeline,
        has_mask=True,
    ),
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< config

if __name__ == "__main__":

    dataset = build_dataset(data['train'])
    data_loader = build_dataloader(
        dataset,
        3,
        0,
        dist=False,
    )
    data_loader = iter(data_loader)
    
    dst_root = osp.splitext(__file__)[0] + '__out'
    os.system(f'rm -rf {dst_root}')

    save_keys = ['img', 'gt_semantic_seg']
    dst_dirs = [osp.join(dst_root, key) for key in save_keys]
    for dst_dir in dst_dirs:
        os.makedirs(dst_dir, exist_ok=True)
    
    for i in range(100):
        sample = next(data_loader)


        # --------- weightmaps
        mmcv.imwrite(
            sample['weightmaps']['gt_semantic_seg'][0].numpy().astype(np.uint8),
            osp.join(dst_root, 'weightmap/gt_semantic_seg', f'{i}.png')
        )

        img1 = sample['img'][0][0].numpy().astype(np.uint8)
        img2 = sample['img'][1][0].numpy().astype(np.uint8)

        mmcv.imwrite(img1, osp.join(dst_root, 'img1', f'{i}.png'))
        mmcv.imwrite(img2, osp.join(dst_root, 'img2', f'{i}.png'))

        # --------- gt
        arr = sample['gt_semantic_seg'][0].numpy().astype(np.uint8)
        arr_color = np.zeros((arr.shape[0], arr.shape[1], 3),
                             dtype=np.uint8)
        for label, color in PALETTE.items():
            arr_color[arr == label, :] = color

        mmcv.imwrite(
            arr_color,
            osp.join(dst_root, 'gt_semantic_seg', f'{i}.png')
        ) 
