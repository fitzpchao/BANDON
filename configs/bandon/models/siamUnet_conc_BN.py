norm_cfg = dict(type='SyncBN', requires_grad=True) #Maybe the implement of SyncBN
model = dict(
    type='CDNet',
    backbone=dict(
        type='SiamUnet_conc',
        use_IN1=False,
        norm_cfg=norm_cfg,
    ),
    decode_head=[
        dict(
            type='CLSHead',
            in_channels=16,
            channels=16,
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
