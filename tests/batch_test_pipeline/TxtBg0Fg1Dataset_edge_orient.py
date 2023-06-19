import os, sys
import os.path as osp
import numpy as np
import cv2
import mmcv
from mmseg.datasets import TxtBg0Fg1Dataset
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.datasets.pipelines import OrientationUtil 
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
dataset_type = 'TxtBg0Fg1Dataset'
data_root = ''
img_norm_cfg = dict(
    mean = [127, 127, 127],
    std  = [58, 58, 58],
    to_rgb=True
)
crop_size = (896, 896)
collect_keys = ['img', 'gt_semantic_seg', 'gt_edge_seg',
    'gt_orient_edge_18_seg',
#   'gt_edge_distance_seg',
    'gt_edge_distance_reg',
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=None, ratio_range=(0.875, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='BinaryMask2Edge'),
    dict(type='BinaryMask2OrientEdge', num_bins=[18], blur_sigma=5),
   #dict(type='BinaryMask2EdgeDistance', clip_min=0, clip_max=1023,
#  #    num_bins=7,
   #    reg_to_cls=False),
    dict(type='Mask2EdgeDistance', clip_min=0, clip_max=1023, ch_ind=[0, 1]),
#   dict(type='Normalize', **img_norm_cfg),
#   dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#   dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=collect_keys),
]
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        txt_fn = '/mnt/lustre/wujiang/GDX/_model_release/' \
            'M_Remote_Segment_Water_4.0.0.model_res__213245.txt',
        pipeline=train_pipeline,
    ),
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< config

ort18 = OrientationUtil(18)
ort36 = OrientationUtil(36)
ort180 = OrientationUtil(180)

if __name__ == "__main__":
    dataset = build_dataset(data['train'])
    data_loader = build_dataloader(
        dataset,
        2,
        0,
        dist=False,
    )
    data_loader = iter(data_loader)
    
    dst_root = osp.splitext(__file__)[0] + '__out'
    os.system(f'rm -rf {dst_root}')
    dst_dirs = [osp.join(dst_root, key) for key in collect_keys]
    for dst_dir in dst_dirs:
        os.makedirs(dst_dir, exist_ok=True)
    
    for i in range(100):
        sample = next(data_loader)

        for key, dst_dir in zip(collect_keys, dst_dirs):
            arr = sample[key][0]
            try:
                arr = sample[key][0]
            except:
                print(f'key = {key}')

            if key == 'gt_edge_distance_reg':
                arr = (arr.numpy() / 4.0).astype(np.uint8)
                print(f'arr.shape = {arr.shape}')
                C = arr.shape[0]
                for c in range(C):
                    mmcv.imwrite(arr[c,:,:], osp.join(dst_dir, f'{c}/{i}.png'))
                
            else:
                arr = arr.numpy().astype(np.uint8)

                # ------ uint8 to color
                if key in {'gt_semantic_seg', 'gt_edge_seg', 'gt_edge_distance_seg'}:
                    arr_color = np.zeros((arr.shape[0], arr.shape[1], 3),
                                         dtype=np.uint8)
                    for label, color in PALETTE.items():
                        arr_color[arr == label, :] = color
                    arr = arr_color
                elif key == 'gt_orient_edge_18_seg':
                    arr_color = ort18.label_to_color(arr)
                    arr_color = cv2.cvtColor(arr_color, cv2.COLOR_BGR2RGB)
                    mmcv.imwrite(arr_color, osp.join(dst_dir+'_color', f'{i}.png'))
                elif key == 'gt_orient_edge_36_seg':
                    arr_color = ort36.label_to_color(arr)
                    arr_color = cv2.cvtColor(arr_color, cv2.COLOR_BGR2RGB)
                    mmcv.imwrite(arr_color, osp.join(dst_dir+'_color', f'{i}.png'))
                elif key == 'gt_orient_edge_180_seg':
                    arr_color = ort180.label_to_color(arr)
                    arr_color = cv2.cvtColor(arr_color, cv2.COLOR_BGR2RGB)
                    mmcv.imwrite(arr_color, osp.join(dst_dir+'_color', f'{i}.png'))
             
                mmcv.imwrite(arr, osp.join(dst_dir, f'{i}.png'))
        sys.stdout.flush()
    
