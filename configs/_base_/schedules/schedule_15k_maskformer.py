# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0)
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
# learning policy
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))


# runtime settings
total_iters = 15000
runner = dict(type='IterBasedRunner', max_iters=total_iters )
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=10000, metric='mIoU')