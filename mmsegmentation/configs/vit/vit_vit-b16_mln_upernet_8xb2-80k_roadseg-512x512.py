_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/roadsegds.py', '../_base_/default_runtime.py',
    #'../_base_/schedules/schedule_25k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    #pretrained='pretrain/vit_base_patch16_224.pth',
    pretrained=None,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=10,
        by_epoch=True)
]
# training schedule for 25k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
