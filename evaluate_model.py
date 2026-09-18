import json
import time
from copy import deepcopy
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner import Runner


# ============================================================
# Hardcoded Paths
# ============================================================

# Project root
PROJECT_ROOT = Path(
    "/home/ezio-gpu/Documents/Vinith/MMDetection_2"
)

# CPU training configuration
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "packdet_rtmdet_ins_cpu.py"
)

# CPU-trained checkpoint
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "work_dirs"
    / "packdet_rtmdet_ins_cpu"
    / "epoch_25.pth"
)

# Dataset root
DATA_ROOT = PROJECT_ROOT / "data"

# Validation COCO annotation file
ANNOTATION_PATH = (
    DATA_ROOT
    / "annotations"
    / "instances_validation.json"
)

# Available image directories
TRAIN_IMAGE_DIR = (
    DATA_ROOT
    / "images"
    / "Train"
)

VALIDATION_IMAGE_DIR = (
    DATA_ROOT
    / "images"
    / "Validation"
)

# Select the image directory used by the validation annotation
#
# Default:
# instances_validation.json -> images/Validation
#
# If your JSON references images from Train,
# change this line to:
#
# IMAGE_DIR = TRAIN_IMAGE_DIR
#
IMAGE_DIR = VALIDATION_IMAGE_DIR

# Evaluation output directory
WORK_DIR = (
    PROJECT_ROOT
    / "evaluation_results"
    / "cpu_model_on_gpu"
)

# Evaluation device
DEVICE = "cuda:0"


# ============================================================
# Helper Functions
# ============================================================

def check_exists(path: Path, description: str):
    """Check whether a file or directory exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )


def print_environment(device: str):
    """Print and validate the evaluation environment."""

    print("\n" + "=" * 75)
    print("EVALUATION ENVIRONMENT")
    print("=" * 75)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Requested device: {device}")

    if device.startswith("cuda"):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        gpu_index = torch.cuda.current_device()

        print(f"GPU index       : {gpu_index}")
        print(
            "GPU name        : "
            f"{torch.cuda.get_device_name(gpu_index)}"
        )

    print("=" * 75)


def update_dataset_paths(
    cfg: Config,
    data_root: Path,
    annotation_path: Path,
    image_dir: Path,
):
    """
    Override the Windows dataset paths from the training config.

    This is required because the original training configuration
    contains Windows paths, while evaluation is running on Linux.
    """

    if "test_dataloader" not in cfg:
        raise RuntimeError(
            "test_dataloader is missing from the configuration."
        )

    if "test_evaluator" not in cfg:
        raise RuntimeError(
            "test_evaluator is missing from the configuration."
        )

    # --------------------------------------------------------
    # Copy and update test dataloader
    # --------------------------------------------------------

    test_dataloader = deepcopy(cfg.test_dataloader)

    dataset = test_dataloader["dataset"]

    dataset["data_root"] = str(data_root.resolve())

    # COCO dataset annotation path relative to data_root
    dataset["ann_file"] = str(
        annotation_path.resolve().relative_to(
            data_root.resolve()
        )
    )

    # Image directory relative to data_root
    dataset["data_prefix"] = {
        "img": str(
            image_dir.resolve().relative_to(
                data_root.resolve()
            )
        )
    }

    dataset["test_mode"] = True

    cfg.test_dataloader = test_dataloader

    # --------------------------------------------------------
    # Update COCO evaluator
    # --------------------------------------------------------

    evaluator = deepcopy(cfg.test_evaluator)

    evaluator["ann_file"] = str(
        annotation_path.resolve()
    )

    cfg.test_evaluator = evaluator

    return cfg


def prepare_config(
    config_path: Path,
    checkpoint_path: Path,
    work_dir: Path,
    data_root: Path,
    annotation_path: Path,
    image_dir: Path,
    device: str,
):
    """Load and prepare the MMDetection evaluation configuration."""

    print("\nLoading configuration:")
    print(config_path)

    cfg = Config.fromfile(str(config_path))

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    cfg.load_from = str(checkpoint_path.resolve())

    # --------------------------------------------------------
    # Test loop
    # --------------------------------------------------------

    # Required because the original config did not define test_cfg
    cfg.test_cfg = dict(
        type="TestLoop"
    )

    # --------------------------------------------------------
    # Runtime settings
    # --------------------------------------------------------

    cfg.device = device
    cfg.launcher = "none"

    if "env_cfg" not in cfg:
        cfg.env_cfg = dict()

    cfg.env_cfg.cudnn_benchmark = False

    # Linux multiprocessing configuration
    cfg.env_cfg.mp_cfg = dict(
        mp_start_method="spawn",
        opencv_num_threads=0,
    )

    # --------------------------------------------------------
    # Work directory
    # --------------------------------------------------------

    cfg.work_dir = str(work_dir.resolve())

    # --------------------------------------------------------
    # Override dataset paths
    # --------------------------------------------------------

    cfg = update_dataset_paths(
        cfg=cfg,
        data_root=data_root,
        annotation_path=annotation_path,
        image_dir=image_dir,
    )

    return cfg


def convert_to_json_serializable(value):
    """Convert MMEngine results into JSON-compatible objects."""

    if isinstance(value, dict):
        return {
            str(key): convert_to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            convert_to_json_serializable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ============================================================
# Evaluation
# ============================================================

def evaluate_model():
    """Run RTMDet-Ins evaluation on the CUDA GPU."""

    print_environment(DEVICE)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    check_exists(
        CONFIG_PATH,
        "Configuration file",
    )

    check_exists(
        CHECKPOINT_PATH,
        "Checkpoint file",
    )

    check_exists(
        DATA_ROOT,
        "Dataset root directory",
    )

    check_exists(
        ANNOTATION_PATH,
        "COCO annotation file",
    )

    check_exists(
        TRAIN_IMAGE_DIR,
        "Training image directory",
    )

    check_exists(
        VALIDATION_IMAGE_DIR,
        "Validation image directory",
    )

    check_exists(
        IMAGE_DIR,
        "Selected evaluation image directory",
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print evaluation information
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("RTMDET-INS MODEL EVALUATION")
    print("=" * 75)

    print(f"Config       : {CONFIG_PATH}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Annotations  : {ANNOTATION_PATH}")
    print(f"Train images : {TRAIN_IMAGE_DIR}")
    print(f"Valid images : {VALIDATION_IMAGE_DIR}")
    print(f"Selected imgs: {IMAGE_DIR}")
    print(f"Work dir     : {WORK_DIR}")
    print(f"Device       : {DEVICE}")

    # --------------------------------------------------------
    # Prepare configuration
    # --------------------------------------------------------

    cfg = prepare_config(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        work_dir=WORK_DIR,
        data_root=DATA_ROOT,
        annotation_path=ANNOTATION_PATH,
        image_dir=IMAGE_DIR,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Save effective evaluation config
    # --------------------------------------------------------

    final_config_path = (
        WORK_DIR / "evaluation_config.py"
    )

    cfg.dump(str(final_config_path))

    print("\nFinal evaluation config saved to:")
    print(final_config_path)

    # --------------------------------------------------------
    # Initialize runner
    # --------------------------------------------------------

    print("\nInitializing MMDetection Runner...")

    runner = Runner.from_cfg(cfg)

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print("\nStarting evaluation...")
    print("Evaluation mode: TEST")
    print("Metrics        : bbox + segm")

    start_time = time.perf_counter()

    results = runner.test()

    elapsed_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED")
    print("=" * 75)

    print(
        f"Evaluation time: "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print("\nEvaluation results:")
    print(results)

    # --------------------------------------------------------
    # Save results as JSON
    # --------------------------------------------------------

    results_path = (
        WORK_DIR / "evaluation_results.json"
    )

    serializable_results = (
        convert_to_json_serializable(results)
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_results,
            file,
            indent=4,
        )

    print("\nResults saved to:")
    print(results_path)

    print("=" * 75)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate_model()