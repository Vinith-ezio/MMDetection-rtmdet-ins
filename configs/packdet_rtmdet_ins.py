# ============================================================
# PackDet RTMDet-Ins GPU Configuration
# With RAM Monitoring and Memory Cleanup Hook
# ============================================================


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = "/home/ezio-gpu/Documents/Vinith/MMDetection_2"

MMDET_ROOT = BASE_DIR + "/mmdetection"

DATA_ROOT = BASE_DIR + "/data"

WORK_DIR = (
    BASE_DIR
    + "/work_dirs/packdet_rtmdet_ins_gpu"
)


# ============================================================
# Custom Imports
# ============================================================

custom_imports = dict(
    imports=[
        "memory_cleanup_hook",
    ],
    allow_failed_imports=False,
)


# ============================================================
# Dataset Configuration
# ============================================================

data_root = DATA_ROOT + "/"

classes = (
    "plastic_bottle",
    "glass_bottle",
    "can",
    "jar",
    "carton_beverage",
    "tube",
)

metainfo = dict(
    classes=classes
)


# ============================================================
# Base Configuration
# ============================================================

_base_ = (
    "../mmdetection/configs/rtmdet/"
    "rtmdet-ins_s_8xb32-300e_coco.py"
)


# ============================================================
# Training Dataset
# ============================================================

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,

    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,

        ann_file=(
            "annotations/instances_train.json"
        ),

        data_prefix=dict(
            img="images/Train/"
        ),
    ),
)


# ============================================================
# Validation Dataset
# ============================================================

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,

    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,

        ann_file=(
            "annotations/instances_validation.json"
        ),

        data_prefix=dict(
            img="images/Validation/"
        ),
    ),
)


# ============================================================
# Test Dataset
# ============================================================

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,

    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,

        ann_file=(
            "annotations/instances_validation.json"
        ),

        data_prefix=dict(
            img="images/Validation/"
        ),
    ),
)


# ============================================================
# Evaluators
# ============================================================

val_evaluator = dict(
    type="CocoMetric",

    ann_file=(
        data_root
        + "annotations/instances_validation.json"
    ),

    metric=[
        "bbox",
        "segm",
    ],
)

test_evaluator = dict(
    type="CocoMetric",

    ann_file=(
        data_root
        + "annotations/instances_validation.json"
    ),

    metric=[
        "bbox",
        "segm",
    ],
)


# ============================================================
# Model Configuration
# ============================================================

model = dict(
    bbox_head=dict(
        num_classes=len(classes)
    ),

    test_cfg=dict(
        # Reduce the number of proposals before NMS.
        # This helps reduce GPU memory usage during evaluation.
        nms_pre=50,

        # Limit the final number of detections per image.
        max_per_img=50,

        # Ignore very low-confidence predictions.
        score_thr=0.05,
    ),
)


# ============================================================
# Training Schedule
# ============================================================

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=25,

    # Validate after every epoch.
    val_interval=1,
)

val_cfg = dict(
    type="ValLoop"
)

test_cfg = dict(
    type="TestLoop"
)


# ============================================================
# Optimizer
# ============================================================

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=0.0001,
        weight_decay=0.05,
    )
)


# ============================================================
# Runtime Configuration
# ============================================================

work_dir = WORK_DIR

default_hooks = dict(
    timer=dict(
        type="IterTimerHook"
    ),

    logger=dict(
        type="LoggerHook",
        interval=50,
    ),

    param_scheduler=dict(
        type="ParamSchedulerHook"
    ),

    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,

        save_best=(
            "coco/segm_mAP"
        ),
    ),

    sampler_seed=dict(
        type="DistSamplerSeedHook"
    ),

    visualization=dict(
        type="DetVisualizationHook",
        draw=False,
    ),
)


# ============================================================
# Custom Memory Hook
# ============================================================

custom_hooks = [
    dict(
        type="MemoryCleanupHook",

        # Run once per epoch.
        interval=1,

        # Monitor CPU RAM.
        collect_cpu=True,

        # Clean CUDA cache.
        clean_cuda=True,

        # Monitor swap usage.
        log_swap=True,
    )
]


# ============================================================
# Logging
# ============================================================

log_processor = dict(
    by_epoch=True,
    window_size=50,
)


# ============================================================
# Environment Configuration
# ============================================================

env_cfg = dict(
    cudnn_benchmark=False,

    mp_cfg=dict(
        # Linux multiprocessing method.
        mp_start_method="fork",

        # Prevent OpenCV from creating extra threads.
        opencv_num_threads=0,
    ),

    dist_cfg=dict(
        backend="nccl",
    ),
)


# ============================================================
# Randomness
# ============================================================

randomness = dict(
    seed=42,
    deterministic=False,
)