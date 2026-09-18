import os
import sys
import time
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
import psutil
import torch

from mmengine.config import Config
from mmengine.runner import Runner


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/home/ezio-gpu/Documents/Vinith/MMDetection_2"
)

MMDET_ROOT = PROJECT_ROOT / "mmdetection"

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "packdet_rtmdet_ins.py"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "work_dirs"
    / "packdet_rtmdet_ins"
    / "epoch_25.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation_results"
)

LOG_DIR = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = LOG_DIR / f"evaluation_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def log_system_information():
    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)

    logger.info("Operating System: %s", platform.platform())
    logger.info("Python Version: %s", platform.python_version())
    logger.info("PyTorch Version: %s", torch.__version__)

    logger.info("CUDA Available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA Version: %s", torch.version.cuda)
        logger.info(
            "GPU Count: %s",
            torch.cuda.device_count(),
        )

        for gpu_index in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_properties = torch.cuda.get_device_properties(
                gpu_index
            )

            total_memory_gb = (
                gpu_properties.total_memory / (1024 ** 3)
            )

            logger.info(
                "GPU %s: %s",
                gpu_index,
                gpu_name,
            )

            logger.info(
                "GPU %s Total Memory: %.2f GB",
                gpu_index,
                total_memory_gb,
            )

    else:
        logger.warning(
            "CUDA is not available. Evaluation will run on CPU."
        )

    ram = psutil.virtual_memory()

    logger.info(
        "Total Physical RAM: %.2f GB",
        ram.total / (1024 ** 3),
    )

    logger.info(
        "Available Physical RAM: %.2f GB",
        ram.available / (1024 ** 3),
    )

    logger.info(
        "Used Physical RAM: %.2f GB",
        ram.used / (1024 ** 3),
    )

    logger.info(
        "Physical RAM Usage: %.2f%%",
        ram.percent,
    )


# ============================================================
# GPU MEMORY INFORMATION
# ============================================================

def get_gpu_memory_information():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

    device_index = torch.cuda.current_device()

    allocated_memory = torch.cuda.memory_allocated(
        device_index
    )

    reserved_memory = torch.cuda.memory_reserved(
        device_index
    )

    max_allocated_memory = torch.cuda.max_memory_allocated(
        device_index
    )

    return {
        "cuda_available": True,
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(
            device_index
        ),
        "allocated_gb": allocated_memory / (1024 ** 3),
        "reserved_gb": reserved_memory / (1024 ** 3),
        "max_allocated_gb": max_allocated_memory / (1024 ** 3),
    }


def log_memory_status(stage):
    ram = psutil.virtual_memory()
    gpu_info = get_gpu_memory_information()

    logger.info("-" * 80)
    logger.info("MEMORY STATUS: %s", stage)
    logger.info("-" * 80)

    logger.info(
        "Physical RAM Used: %.2f GB / %.2f GB",
        ram.used / (1024 ** 3),
        ram.total / (1024 ** 3),
    )

    logger.info(
        "Physical RAM Usage: %.2f%%",
        ram.percent,
    )

    if gpu_info["cuda_available"]:
        logger.info(
            "GPU: %s",
            gpu_info["device_name"],
        )

        logger.info(
            "GPU Allocated Memory: %.2f GB",
            gpu_info["allocated_gb"],
        )

        logger.info(
            "GPU Reserved Memory: %.2f GB",
            gpu_info["reserved_gb"],
        )

        logger.info(
            "GPU Maximum Allocated Memory: %.2f GB",
            gpu_info["max_allocated_gb"],
        )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():
    logger.info("=" * 80)
    logger.info("VALIDATING FILE PATHS")
    logger.info("=" * 80)

    logger.info("Project Root: %s", PROJECT_ROOT)
    logger.info("MMDetection Root: %s", MMDET_ROOT)
    logger.info("Config File: %s", CONFIG_FILE)
    logger.info("Checkpoint File: %s", CHECKPOINT_FILE)
    logger.info("Output Directory: %s", OUTPUT_DIR)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist: {PROJECT_ROOT}"
        )

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {CONFIG_FILE}"
        )

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {CHECKPOINT_FILE}"
        )

    logger.info("All required paths are valid.")


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_evaluation_config():
    logger.info("=" * 80)
    logger.info("LOADING MMDETECTION CONFIGURATION")
    logger.info("=" * 80)

    cfg = Config.fromfile(str(CONFIG_FILE))

    logger.info("Configuration loaded successfully.")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do not use:
    # del cfg["Path"]
    #
    # MMEngine Config does not support __delitem__.
    #
    # The original Path issue is handled by avoiding cfg.dump().
    # --------------------------------------------------------

    if "Path" in cfg:
        logger.warning(
            "The config contains pathlib.Path. "
            "It will be ignored during evaluation."
        )

    # --------------------------------------------------------
    # Select evaluation device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        cfg.device = "cuda:0"

        logger.info(
            "Evaluation device set to CUDA:0"
        )

        logger.info(
            "GPU selected: %s",
            torch.cuda.get_device_name(0)
        )

    else:
        cfg.device = "cpu"

        logger.warning(
            "CUDA unavailable. Evaluation device set to CPU."
        )

    # --------------------------------------------------------
    # Set checkpoint
    # --------------------------------------------------------

    cfg.load_from = str(CHECKPOINT_FILE)

    logger.info(
        "Checkpoint configured: %s",
        CHECKPOINT_FILE
    )

    # --------------------------------------------------------
    # Set evaluation output directory
    # --------------------------------------------------------

    cfg.work_dir = str(OUTPUT_DIR)

    # --------------------------------------------------------
    # Save only safe evaluation metadata
    #
    # Do not use cfg.dump() because the config contains:
    #
    # Path=<class 'pathlib.Path'>
    # --------------------------------------------------------

    config_summary = {
        "config_file": str(CONFIG_FILE),
        "checkpoint_file": str(CHECKPOINT_FILE),
        "device": str(cfg.device),
        "work_dir": str(cfg.work_dir),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "timestamp": timestamp,
    }

    config_summary_file = (
        OUTPUT_DIR
        / f"evaluation_config_{timestamp}.json"
    )

    with open(
        config_summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_summary,
            file,
            indent=4,
        )

    logger.info(
        "Evaluation configuration summary saved to: %s",
        config_summary_file,
    )

    return cfg


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():
    logger.info("=" * 80)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 80)

    evaluation_start_time = time.perf_counter()

    log_memory_status("Before Runner Creation")

    cfg = load_evaluation_config()

    log_memory_status("After Config Loading")

    logger.info("Creating MMEngine Runner...")

    runner = Runner.from_cfg(cfg)

    logger.info("MMEngine Runner created successfully.")

    log_memory_status("After Runner Creation")

    logger.info("Loading checkpoint: %s", CHECKPOINT_FILE)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoint_load_start = time.perf_counter()

    runner.load_or_resume()

    checkpoint_load_end = time.perf_counter()

    logger.info(
        "Checkpoint loading completed in %.2f seconds.",
        checkpoint_load_end - checkpoint_load_start,
    )

    log_memory_status("After Checkpoint Loading")

    logger.info("Starting test/evaluation process...")

    test_start_time = time.perf_counter()

    results = runner.test()

    test_end_time = time.perf_counter()

    logger.info(
        "Evaluation completed in %.2f seconds.",
        test_end_time - test_start_time,
    )

    log_memory_status("After Evaluation")

    evaluation_end_time = time.perf_counter()

    total_evaluation_time = (
        evaluation_end_time - evaluation_start_time
    )

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80)

    logger.info(
        "Total Evaluation Time: %.2f seconds",
        total_evaluation_time,
    )

    logger.info("Evaluation Results:")

    if results is not None:
        logger.info("%s", results)

        results_file = (
            OUTPUT_DIR
            / f"evaluation_results_{timestamp}.json"
        )

        try:
            with open(
                results_file,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    results,
                    file,
                    indent=4,
                    default=str,
                )

            logger.info(
                "Evaluation results saved to: %s",
                results_file,
            )

        except Exception as error:
            logger.warning(
                "Could not save results as JSON: %s",
                error,
            )

    else:
        logger.info(
            "The runner did not return a result dictionary."
        )

    logger.info(
        "Evaluation log saved to: %s",
        LOG_FILE,
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        log_system_information()
        validate_paths()
        run_evaluation()

    except KeyboardInterrupt:
        logger.warning(
            "Evaluation interrupted by user."
        )

    except Exception as error:
        logger.exception(
            "Evaluation failed with error: %s",
            error,
        )

        raise


if __name__ == "__main__":
    main()