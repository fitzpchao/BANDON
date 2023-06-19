# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))




optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
total_epochs = 400
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs )
checkpoint_config = dict(by_epoch=True, interval=50)
evaluation = dict(interval=160, metric='mIoU')
