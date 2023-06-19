# optimizer
optimizer_config = dict()
optimizer = dict(
    type='AdamW',
    lr=0.00006,
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
total_epochs = 800
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs )
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=160, metric='mIoU')
