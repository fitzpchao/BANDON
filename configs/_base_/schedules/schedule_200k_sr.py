# optimizer
optimizer_config = dict()
optimizer = dict(
    type='Adam',
    lr=4e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

lr_config = dict(
    policy='step',
    step=[50000, 100000, 150000],
    gamma=0.5,
    min_lr=0.0,
    by_epoch=False)
# learning policy



# runtime settings
total_iters = 200000
runner = dict(type='IterBasedRunner', max_iters=total_iters )
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=10000, metric='mIoU')