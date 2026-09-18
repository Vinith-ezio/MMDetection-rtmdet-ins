# RTMDet-Ins-S Training

This package contains the RTMDet-Ins-S training configuration and DDP training launcher.

## Files

```text
MMDetection_2/
│
├── train_2.py
├── requirements.txt
├── README.md
├── configs/
│   └── packdet_rtmdet_ins.py
├── data/
│   ├── images/
│   └── annotations/
├── mmdetection/
└── work_dirs/
```

### Main files

* `train_2.py` — Training launcher with GPU check and PyTorch DDP.
* `requirements.txt` — Required Python packages and tested package versions.
* `configs/packdet_rtmdet_ins.py` — RTMDet-Ins-S model and training configuration.
* `data/` — Training dataset and COCO annotations.
* `work_dirs/` — Training logs and checkpoints.

## Current Training Setup

* Model: RTMDet-Ins-S
* GPU: 2 × NVIDIA RTX 3070 8GB
* Distributed training: PyTorch DDP
* Batch size: 2 per GPU
* Effective batch size: 4
* Image size: 640 × 640
* DataLoader workers: 0
* Validation: Disabled during training
* Current training: 5 epochs

## Environment

The required Python packages are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

The current configuration contains Linux-specific paths, so Windows users will need to update the paths/environment before running.

## Run Training

From the project directory:

```bash
python train_2.py
```

The launcher automatically checks the GPU/CUDA environment and starts DDP training.

## Output

Training outputs are saved under:

```text
work_dirs/packdet_rtmdet_ins/
```

This includes:

* Training logs
* Checkpoints (`.pth`)
* Latest checkpoint information

##
