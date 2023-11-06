# dataset settings
dataset_type = 'RoadSegDataset'
data_root = '../../input/dataset-road/RoadSeg'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type="Albu",
         transforms=[
            dict(
                type='HorizontalFlip',
                p=0.5
            ),
            dict(
                type='ShiftScaleRotate',
                scale_limit=0.5, 
                rotate_limit=0, 
                shift_limit=0.1, 
                border_mode=0,
                p=1, 
            ),
            dict(
                type='IAAAdditiveGaussianNoise',
                p=0.2
            ),
            dict(
                type='IAAPerspective',
                p=0.5
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='IAASharpen',
                         p=1),
                    dict(type='Blur',
                         blur_limit=3,
                         p=1),
                    dict(type='MotionBlur',
                         blur_limit=3,
                         p=1)
                ],
                p=0.9
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomContrast',
                         p=1),
                    dict(type='HueSaturationValue',
                         p=1)
                ],
                p=0.9
            )
        ]),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=0., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='WeightedSampler', 
                 images_dir= data_root+'/'+'Train/Image',
                 masks_dir= data_root+'/'+'Train/Mask'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='Train/Image', seg_map_path='Train/Mask'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='Val/Image',
            seg_map_path='Val/Mask'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='Test/Image',
            seg_map_path='Test/Mask'),
        pipeline=tta_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
