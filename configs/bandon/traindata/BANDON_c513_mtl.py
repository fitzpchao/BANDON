# dataset settings
dataset_type = 'TxtMIMODatasetForBANDON'
data_root = ''
img_norm_cfg = dict(
    mean = [0, 0, 0],
    std  = [255, 255, 255],
    to_rgb=True
)
crop_size = (513, 513)
resize_ratio_range=(0.7, 1.5)
pair_offset=0
file_client_args = dict(
    backend='disk',)
train_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='LoadAnnotationsWithMultiLabels',file_client_args=file_client_args,in_keys=['seg_build','flow_build','flow_cd']),
    # dict(type='Resize', img_scale=None, ratio_range=resize_ratio_range),
    # dict(type='RandomRotate', rotate_range=(-180, 180)),
    dict(type='RandomCrop', crop_size=tuple(crop + 2*pair_offset for crop in crop_size)),
    # dict(type='CenterCrop', crop_size=crop_size, rand_pair_trans_offset=pair_offset),
    dict(type='RandomGaussianBlur', blur_ratio=0.4, radius=5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal'),
    dict(type="RandomRotate90n"),
    dict(type='RandomChannelShiftScale',  max_color_shift=20, contrast_range=(0.8, 1.2)),
    dict(type='RandomPixelNoise', max_pixel_noise=20),
    # dict(type='Reg2ClsXY',ths=[-17,-8,-3,0,-3,8,17],in_keys=['gt_flow_cd_reg']),
    dict(type='Reg2ClsXYWithSeg',ths=[-21,-9,-4,0,4,9,21],in_keys=['gt_flow_cd_reg']),
    dict(type='Reg2ClsXYWithSeg', ths=[-13, -5, -2, 0, 2, 5, 13],in_keys=['gt_flow_build_reg_1',
                                                                   'gt_flow_build_reg_2']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_build_seg_1','gt_build_seg_2',
              'gt_flow_build_reg_1_x', 'gt_flow_build_reg_2_x',
              'gt_flow_cd_reg_x'
              ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleAug',
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root = '/remote-home/pangchao/data/BANDON/train',
        txt_fn='lists/list_BANDON_train.txt',
        has_mask=True),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='/remote-home/pangchao/data/BANDON/val',
        txt_fn='lists/list_BANDON_val.txt',
        has_mask=True),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root='/remote-home/pangchao/data/BANDON/test',
        txt_fn='lists/list_BANDON_test.txt',
        has_mask=True)


)
train_cfg = dict()
test_cfg = dict(
    mode='slide',
    crop_size=crop_size,
    stride=(337, 337),
)
