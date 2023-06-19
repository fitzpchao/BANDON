norm_cfg = dict(type='SyncBN', requires_grad=True) #Maybe the implement of SyncBN
model = dict(
    type='CDNet',
    backbone=dict(
        type='Siam_diff_r50',
        depth=50,
        use_IN1=False,
        pretrained='/mnt/lustre/pangchao2/.cache/torch/hub/checkpoints/' \
                   + 'resnet50_v1c-2cccc1ad.pth',
        base_channels=64,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 1, 1),
        deep_stem=True,
        norm_cfg=norm_cfg,
        norm_eval=False,
        contract_dilation=True,
        style='pytorch',
    ),
    decode_head=[
        dict(
            type='CLSHead',
            in_channels=64*4,
            channels=64*4,
            norm_cfg=norm_cfg,
            num_classes=2,
            dropout_ratio=0.1,
            in_index=-1,
            loss_decode=dict(
                type='DiceWithCELoss', class_weight_ce=[1., 2.],
                use_sigmoid=False, loss_weight=1.0
            ),
        )
    ],
    gt_keys = ['gt_semantic_seg']
)
