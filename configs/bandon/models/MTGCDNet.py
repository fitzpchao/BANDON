norm_cfg = dict(type='SyncBN', requires_grad=True) #Maybe the implement of SyncBN
model = dict(
    type='CDNet',
    backbone=dict(
        type='ChangePSPNetMTL',
        depth=50,
        use_IN1=True,
        pretrained='/remote-home/pangchao/checkpoints/resnet50_v1c-2cccc1ad.pth',
        base_channels=64,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        deep_stem=True,
        norm_cfg=norm_cfg,
        norm_eval=False,
        contract_dilation=True,
        style='pytorch',
    ),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            channels=512,
            num_convs=2,
            norm_cfg=norm_cfg,
            num_classes=2,
            dropout_ratio=0.1,
            in_index=0,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='DiceWithCELoss', class_weight_ce=[1., 2.],
                use_sigmoid=False, loss_weight=1.0
            ),
        ),

        dict(
            type='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=1,
            norm_cfg=norm_cfg,
            num_classes=3,
            dropout_ratio=0.1,
            in_index=1,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='CrossEntropyLoss', class_weight=[1., 2., 4.],
                use_sigmoid=False, loss_weight=1.0
            ),
        ),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=1,
            norm_cfg=norm_cfg,
            num_classes=3,
            dropout_ratio=0.1,
            in_index=2,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='CrossEntropyLoss', class_weight=[1., 2., 4.],
                use_sigmoid=False, loss_weight=1.0
            ),
        ),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=1,
            norm_cfg=norm_cfg,
            num_classes=10,
            dropout_ratio=0.1,
            in_index=3,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False, loss_weight=1.0
            )
        ),
    dict(
            type='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=1,
            norm_cfg=norm_cfg,
            num_classes=10,
            dropout_ratio=0.1,
            in_index=4,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False, loss_weight=1.0
            )
        ),
    dict(
            type='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=1,
            norm_cfg=norm_cfg,
            num_classes=10,
            dropout_ratio=0.1,
            in_index=5,
            align_corners=True,
            concat_input=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False, loss_weight=1.0
            )
        ),

    ],
    gt_keys = ['gt_semantic_seg','gt_build_seg_1','gt_build_seg_2','gt_flow_cd_reg_x','gt_flow_build_reg_1_x', 'gt_flow_build_reg_2_x',
              ]
)