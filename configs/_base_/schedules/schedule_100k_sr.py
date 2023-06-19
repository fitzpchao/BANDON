# optimizer
optimizer_config = dict()
optimizer = dict(
    type='Adam',
    lr=2e-5,
    betas=(0.9, 0.99),
)

lr_config = dict(
    policy='step',
    step=[50000, 90000],
    gamma=0.5,
    min_lr=0.0,
    by_epoch=False)
# learning policy

optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))


# runtime settings
total_iters = 100000
runner = dict(type='IterBasedRunner', max_iters=total_iters )
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=10000, metric='mIoU')