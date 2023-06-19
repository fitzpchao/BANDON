norm_cfg = dict(type='SyncBN', requires_grad=True) #Maybe the implement of SyncBN
model = dict(
    type='CDNet',
    backbone=dict(
        type='ChangeFormerV6',
        input_nc=3,
        img_size=512,
        embed_dim=256,
        pretrained='/remote-home/pangchao/checkpoints/segformer.b2.512x512.ade.160k.pc.pth',
        conv_cfg=None,
        norm_cfg=norm_cfg,
    ),
    decode_head=[
        dict(
            type='CLSHead',
            in_channels=256,
            channels=256,
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
find_unused_parameters=True