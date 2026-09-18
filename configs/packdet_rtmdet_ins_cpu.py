# ============================================================
# PACKDET - RTMDET-INS-S CONFIGURATION
# Windows + CPU RAM Monitoring Training
# ============================================================


# ============================================================
# BASE CONFIGURATION
# ============================================================

_base_ = (
    r'../mmdetection/configs/rtmdet/'
    r'rtmdet-ins_s_8xb32-300e_coco.py'
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Windows project directory
BASE_DIR = r'E:\MMDetection_2'

# Dataset root
data_root = BASE_DIR + r'\data\\'

# Training output directory
work_dir = BASE_DIR + r'\work_dirs\packdet_rtmdet_ins'


# ============================================================
# CUSTOM IMPORTS
# ============================================================

custom_imports = dict(
    imports=[
        'mmdet',
    ],
    allow_failed_imports=False,
)


# ============================================================
# DATASET CLASSES
# ============================================================

classes = (
    'plastic_bottle',
    'glass_bottle',
    'can',
    'jar',
    'carton_beverage',
    'tube',
)

metainfo = dict(
    classes=classes,
)


# ============================================================
# MODEL
# ============================================================

model = dict(
    bbox_head=dict(
        num_classes=len(classes),
    ),

    test_cfg=dict(
        # Number of proposals before NMS
        nms_pre=100,

        # Minimum bounding-box size
        min_bbox_size=0,

        # Detection confidence threshold
        score_thr=0.05,

        # Non-maximum suppression
        nms=dict(
            type='nms',
            iou_threshold=0.6,
        ),

        # Maximum detections per image
        max_per_img=20,

        # Mask threshold
        mask_thr_binary=0.5,
    ),
)


# ============================================================
# TRAINING PIPELINE
# ============================================================

train_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
    ),

    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
    ),

    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114),
        ),
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
    ),

    dict(
        type='PackDetInputs',
    ),
]


# ============================================================
# VALIDATION / TEST PIPELINE
# ============================================================

test_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
    ),

    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
    ),

    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114),
        ),
    ),

    dict(
        type='PackDetInputs',
    ),
]


# ============================================================
# TRAINING DATASET
# ============================================================

train_dataset = dict(
    type='CocoDataset',

    data_root=data_root,

    ann_file='annotations/instances_train.json',

    data_prefix=dict(
        img='images/Train/',
    ),

    metainfo=metainfo,

    filter_cfg=dict(
        filter_empty_gt=True,
        min_size=32,
    ),

    pipeline=train_pipeline,
)


# ============================================================
# TRAINING DATALOADER
# ============================================================

train_dataloader = dict(
    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    pin_memory=False,

    dataset=train_dataset,
)


# ============================================================
# VALIDATION DATALOADER
#
# Validation is disabled for the current RAM test.
# ============================================================

val_dataloader = None


# ============================================================
# TEST DATALOADER
#
# This uses the validation COCO JSON.
#
# The image directory is currently set to images/Train/
# because your command confirmed that directory exists.
#
# If a separate images/Validation/ directory exists,
# change this to:
#
#     img='images/Validation/'
# ============================================================

test_dataloader = dict(
    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    pin_memory=False,

    dataset=dict(
        type='CocoDataset',

        data_root=data_root,

        ann_file='annotations/instances_validation.json',

        data_prefix=dict(
            img='images/Train/',
        ),

        metainfo=metainfo,

        test_mode=True,

        pipeline=test_pipeline,
    ),
)


# ============================================================
# VALIDATION EVALUATOR
#
# Disabled because val_cfg=None.
# ============================================================

val_evaluator = None


# ============================================================
# TEST EVALUATOR
# ============================================================

test_evaluator = dict(
    type='CocoMetric',

    ann_file=data_root + r'annotations\instances_validation.json',

    metric=[
        'bbox',
        'segm',
    ],
)


# ============================================================
# TRAINING LOOP
# ============================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',

    max_epochs=25,

    # Does not matter while validation is disabled
    val_interval=5,
)


# ============================================================
# DISABLE VALIDATION
#
# This is useful for the RAM monitoring test.
# Training will run without validation after each epoch.
# ============================================================

val_cfg = None


# ============================================================
# OPTIMIZER
# ============================================================

optim_wrapper = dict(
    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',

        lr=0.0005,

        weight_decay=0.05,
    ),

    clip_grad=dict(
        max_norm=35,

        norm_type=2,
    ),
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

param_scheduler = [
    dict(
        type='CosineAnnealingLR',

        T_max=25,

        by_epoch=True,

        begin=0,

        end=25,
    ),
]


# ============================================================
# CHECKPOINT AND LOGGING HOOKS
# ============================================================

default_hooks = dict(
    logger=dict(
        type='LoggerHook',

        interval=50,
    ),

    checkpoint=dict(
        type='CheckpointHook',

        # Save checkpoint every epoch
        interval=1,

        # Keep only the latest checkpoint
        max_keep_ckpts=1,

        # Save final checkpoint
        save_last=True,
    ),

    visualization=dict(
        type='DetVisualizationHook',

        # Do not save visualization images
        draw=False,

        interval=1,
    ),
)


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

env_cfg = dict(
    cudnn_benchmark=False,

    mp_cfg=dict(
        mp_start_method='spawn',
        opencv_num_threads=0,
    ),

    dist_cfg=dict(
        backend='gloo',
    ),
)


# ============================================================
# RANDOMNESS
# ============================================================

randomness = dict(
    seed=42,

    deterministic=False,
)


# ============================================================
# RESUME SETTINGS
# ============================================================

resume = False


# ============================================================
# LOGGING SETTINGS
# ============================================================

log_processor = dict(
    type='LogProcessor',

    window_size=50,

    by_epoch=True,
)


# ============================================================
# VISUALIZATION BACKEND
# ============================================================

vis_backends = [
    dict(
        type='LocalVisBackend',
    ),
]


visualizer = dict(
    type='DetLocalVisualizer',

    vis_backends=vis_backends,

    name='visualizer',
)


# ============================================================
# TRAINING OUTPUT
# ============================================================

work_dir = work_dir