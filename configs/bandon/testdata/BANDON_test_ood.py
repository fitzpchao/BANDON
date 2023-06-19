dataset_type = 'TxtMIMODatasetForBANDON'
data_root = ''
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
classes=[
    'unchange',
    'change',
]
palette=[
    [0  , 0  , 0  ],
    [255, 255, 255],
]
file_client_args = dict(
    backend='disk'
    )
test_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(
        type='MultiScaleAug_RS',
        img_ratios=[1.0],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        type=dataset_type,
        txt_fn='./lists/list_BANDON_test_ood.txt',
        pipeline=test_pipeline,
        data_root='/remote-home/pangchao/data/BANDON/test_ood',
        has_mask=True,
        classes=classes,
        palette=palette,
    ),
)
