# optimizer
optimizer = dict(type='SGD', lr=1e-1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4,by_epoch=False)

# runtime settings
total_epochs = 400
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs )
checkpoint_config = dict(by_epoch=True, interval=50)
evaluation = dict(interval=160, metric='mIoU')
