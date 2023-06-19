# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, by_epoch=False)
# runtime settings
total_epochs = 30
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=160, metric='mIoU')
