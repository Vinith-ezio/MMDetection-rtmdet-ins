# MMDetection 2.1 --- RTMDet-Ins Scripts

This README documents the Python scripts used in the RTMDet-Ins
detection and instance-segmentation workflow.

## Script Overview

  -----------------------------------------------------------------------
  Script                              Purpose
  ----------------------------------- -----------------------------------
  `train_2.py`                        Starts and manages RTMDet-Ins model
                                      training

  `evaluate_model.py`                 Evaluates a trained model using the
                                      configured evaluation pipeline

  `evaluate_gpu_model.py`             Evaluates a trained model using GPU
                                      execution

  `memory_cleanup_hook.py`            Helps release unused Python and
                                      CUDA memory during execution

  `check_packdet_labels.py`           Validates PackDet annotation files

  `convert_packdet_to_coco.py`        Converts PackDet annotations into
                                      COCO format
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 1. `train_2.py`

### Purpose

`train_2.py` is the main training script for the RTMDet-Ins model.

### Main responsibilities

-   Loads the MMDetection training configuration.
-   Sets the project and dataset paths.
-   Loads the PackDet dataset.
-   Initializes the RTMDet-Ins model.
-   Starts model training.
-   Saves model checkpoints.
-   Generates training logs.
-   Supports resuming training from a checkpoint when configured.
-   Works with the configured CPU/GPU environment.

### Main inputs

-   RTMDet-Ins configuration file.
-   Training images.
-   COCO-format training annotations.
-   Validation images.
-   COCO-format validation annotations.
-   Optional pretrained checkpoint.

### Main outputs

Training outputs are stored inside the configured `work_dir`.

Typical outputs include:

``` text
work_dirs/
└── packdet_rtmDet_ins/
    ├── epoch_1.pth
    ├── epoch_2.pth
    ├── latest.pth
    ├── config.py
    ├── training logs
    └── visualization data
```

### Typical execution

``` powershell
python train_2.py
```

If the script requires a configuration argument:

``` powershell
python train_2.py --config configs/packdet/rtmdet_ins_packdet.py
```

------------------------------------------------------------------------

## 2. `evaluate_model.py`

### Purpose

`evaluate_model.py` evaluates a trained RTMDet-Ins checkpoint against
the validation or test dataset.

### Main responsibilities

-   Loads the model configuration.
-   Loads a trained `.pth` checkpoint.
-   Loads the selected dataset annotations.
-   Runs inference on the dataset.
-   Calculates detection and segmentation metrics.
-   Saves evaluation results.
-   Helps compare different checkpoints.

### Evaluation outputs

Typical results may include:

``` text
evaluation_results/
├── validation/
│   ├── metrics.json
│   ├── bbox_results.json
│   ├── segm_results.json
│   └── validation_log.txt
│
└── testing/
    ├── test_metrics.json
    ├── test_predictions.json
    └── test_log.txt
```

### Metrics

The evaluation process can report:

-   Bounding-box mAP.
-   Bounding-box AP50.
-   Bounding-box AP75.
-   Segmentation mAP.
-   Segmentation AP50.
-   Segmentation AP75.
-   Per-class evaluation results.

### Typical execution

``` powershell
python evaluate_model.py
```

If arguments are supported:

``` powershell
python evaluate_model.py `
    --config configs/packdet/rtmdet_ins_packdet.py `
    --checkpoint work_dirs/packdet_rtmDet_ins/latest.pth
```

------------------------------------------------------------------------

## 3. `evaluate_gpu_model.py`

### Purpose

`evaluate_gpu_model.py` is the GPU-based evaluation script for testing a
trained RTMDet-Ins model.

### Main responsibilities

-   Selects the CUDA device when available.
-   Loads the RTMDet-Ins configuration.
-   Loads the trained checkpoint.
-   Runs validation or testing inference on the GPU.
-   Calculates detection and segmentation metrics.
-   Saves the evaluation results.
-   Helps compare CPU and GPU evaluation behavior.

### Main inputs

-   Model configuration.
-   Trained checkpoint.
-   Dataset images.
-   COCO annotation file.
-   CUDA-enabled PyTorch environment.

### Typical execution

``` powershell
python evaluate_gpu_model.py
```

Example with configuration and checkpoint:

``` powershell
python evaluate_gpu_model.py `
    --config configs/packdet/rtmdet_ins_packdet.py `
    --checkpoint work_dirs/packdet_rtmDet_ins/latest.pth
```

### GPU memory considerations

GPU evaluation may require significant memory because of:

-   High-resolution input images.
-   Batch processing.
-   Detection proposals.
-   Mask generation.
-   Mask post-processing.
-   Large validation datasets.

If CUDA out-of-memory errors occur, reduce the evaluation batch size and
review the model's inference settings.

------------------------------------------------------------------------

## 4. `memory_cleanup_hook.py`

### Purpose

`memory_cleanup_hook.py` is used to reduce unnecessary memory retention
during training or evaluation.

### Main responsibilities

-   Performs Python garbage collection.
-   Releases unused CUDA cache when CUDA is available.
-   Helps reduce memory retained by temporary tensors.
-   Can be integrated into the training or evaluation workflow.
-   Supports investigation of physical RAM and GPU memory usage.

### Main techniques

Typical memory-cleanup operations include:

``` python
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Important note

Memory cleanup does not reduce the memory required by the model itself.
It mainly helps release memory that is no longer needed but is still
retained by Python or the CUDA allocator.

------------------------------------------------------------------------

## 5. `check_packdet_labels.py`

### Purpose

`check_packdet_labels.py` validates the original PackDet annotation
files before conversion or training.

### Main responsibilities

-   Scans the annotation directory.
-   Counts annotation files.
-   Counts polygon objects.
-   Detects invalid annotation lines.
-   Checks annotation formatting.
-   Helps identify corrupted or incomplete labels.

### Example validation output

``` text
Total label files: 722
Total polygon objects: 9086
Invalid annotation lines: 0
```

### Typical execution

``` powershell
python check_packdet_labels.py
```

------------------------------------------------------------------------

## 6. `convert_packdet_to_coco.py`

### Purpose

`convert_packdet_to_coco.py` converts PackDet polygon annotations into
COCO-format JSON files.

### Main responsibilities

-   Reads PackDet image and annotation files.
-   Converts polygon annotations into COCO segmentation format.
-   Creates image records.
-   Creates annotation records.
-   Creates category records.
-   Generates training and validation JSON files.
-   Makes the dataset compatible with MMDetection.

### Generated files

``` text
data/
└── annotations/
    ├── instances_train.json
    ├── instances_validation.json
    └── instances_validation_subset.json
```

### Typical COCO categories

``` text
plastic_bottle
glass_bottle
can
jar
carton_beverage
tube
```

### Typical execution

``` powershell
python convert_packdet_to_coco.py
```

------------------------------------------------------------------------

## Script Execution Order

The normal workflow is:

``` text
1. check_packdet_labels.py
        │
        ▼
2. convert_packdet_to_coco.py
        │
        ▼
3. train_2.py
        │
        ▼
4. evaluate_model.py
   or
   evaluate_gpu_model.py
        │
        ▼
5. Review metrics, predictions and memory logs
```

------------------------------------------------------------------------

## Recommended Script Organization

``` text
MMDetection_2.1/
│
├── scripts/
│   ├── dataset/
│   │   ├── check_packdet_labels.py
│   │   └── convert_packdet_to_coco.py
│   │
│   ├── training/
│   │   └── train_2.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── evaluate_gpu_model.py
│   │
│   └── monitoring/
│       └── memory_cleanup_hook.py
│
├── configs/
├── data/
├── evaluation_results/
├── work_dirs/
└── mmdetection/
```

------------------------------------------------------------------------

## Common Command Examples

### Validate annotations

``` powershell
python scripts/dataset/check_packdet_labels.py
```

### Convert annotations

``` powershell
python scripts/dataset/convert_packdet_to_coco.py
```

### Train the model

``` powershell
python scripts/training/train_2.py
```

### Evaluate on CPU or configured device

``` powershell
python scripts/evaluation/evaluate_model.py
```

### Evaluate using GPU

``` powershell
python scripts/evaluation/evaluate_gpu_model.py
```

------------------------------------------------------------------------

## Important Configuration Values

The scripts may use the following configuration values:

``` python
BASE_DIR = r"/home/ezio-gpu/Documents/Vinith/MMDetection_2"
data_root = BASE_DIR + r"/data/"
work_dir = BASE_DIR + r"/work_dirs/packdet_rtmdet_ins"
```

The actual paths must be changed according to the machine where the
project is executed.

The model configuration uses six classes:

``` python
classes = (
    "plastic_bottle",
    "glass_bottle",
    "can",
    "jar",
    "carton_beverage",
    "tube",
)
```

The RTMDet-Ins bounding-box head must use:

``` python
num_classes = len(classes)
```

------------------------------------------------------------------------

## Troubleshooting Notes

### `NotADirectoryError` or missing dataset files

Check:

-   `data_root`.
-   Image directory.
-   Annotation JSON path.
-   Dataset type.
-   Relative versus absolute paths.

### CUDA out-of-memory error

Check:

-   Batch size.
-   Image resolution.
-   Number of DataLoader workers.
-   Number of proposals.
-   Mask post-processing.
-   GPU memory usage.

### High physical RAM usage

Check:

-   DataLoader worker count.
-   Dataset caching.
-   Large image resolution.
-   Repeated evaluation without releasing objects.
-   Retained prediction results.
-   Python garbage collection.
-   Memory logs per epoch.

### Checkpoint mismatch

Ensure that the following belong to the same experiment:

-   Configuration file.
-   Dataset classes.
-   Dataset annotations.
-   Checkpoint.
-   Number of model classes.
-   Model architecture.

------------------------------------------------------------------------

## Project Script Status

  Script                         Role
  ------------------------------ ------------------------------
  `train_2.py`                   Main RTMDet-Ins training
  `evaluate_model.py`            Model evaluation
  `evaluate_gpu_model.py`        GPU model evaluation
  `memory_cleanup_hook.py`       Memory cleanup and debugging
  `check_packdet_labels.py`      Annotation validation
  `convert_packdet_to_coco.py`   PackDet-to-COCO conversion

This README describes the purpose and expected workflow of the project
scripts. Exact command-line arguments depend on the implementation
inside each Python file.
