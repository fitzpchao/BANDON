# optimizer
optimizer_config = dict()
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'pos_block': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         'head': dict(lr_mult=10.)
    #     })
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



# runtime settings
total_iters = 40000
runner = dict(type='IterBasedRunner', max_iters=total_iters )
checkpoint_config = dict(by_epoch=False, interval=12000)
evaluation = dict(interval=10000, metric='mIoU')