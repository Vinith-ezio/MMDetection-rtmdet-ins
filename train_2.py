# ============================================================
# RTMDet-Ins Training Launcher
# With RAM and GPU Memory Monitoring
# ============================================================

import os
import sys
import time
import signal
import threading
import subprocess

from pathlib import Path
from datetime import datetime


# ============================================================
# Configuration
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

WORK_DIR = (
    PROJECT_ROOT
    / "work_dirs"
    / "packdet_rtmdet_ins"
)

LOG_DIR = WORK_DIR / "system_logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Environment Configuration
# ============================================================

os.environ["PYTHONUNBUFFERED"] = "1"

# Limit CPU thread usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Log File Configuration
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

training_log = (
    LOG_DIR
    / f"training_{timestamp}.log"
)

memory_log = (
    LOG_DIR
    / f"memory_{timestamp}.log"
)


# ============================================================
# Global Monitor Control
# ============================================================

stop_monitor_event = threading.Event()


# ============================================================
# Logging Function
# ============================================================

def log(message, file_path=None):
    """
    Print a timestamped message to the terminal
    and optionally write it to a log file.
    """

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    final_message = (
        f"[{current_time}] {message}"
    )

    print(
        final_message,
        flush=True,
    )

    if file_path is not None:
        with open(
            file_path,
            "a",
            encoding="utf-8",
            buffering=1,
        ) as file:

            file.write(
                final_message + "\n"
            )

            file.flush()


# ============================================================
# System RAM Monitoring
# ============================================================

def get_system_memory():
    """
    Read system RAM, process RSS, and swap usage.
    """

    try:
        import psutil

        process = psutil.Process(
            os.getpid()
        )

        process_memory = (
            process.memory_info()
        )

        system_memory = (
            psutil.virtual_memory()
        )

        swap_memory = (
            psutil.swap_memory()
        )

        return {
            "process_rss_gb": (
                process_memory.rss
                / (1024 ** 3)
            ),

            "ram_total_gb": (
                system_memory.total
                / (1024 ** 3)
            ),

            "ram_used_gb": (
                system_memory.used
                / (1024 ** 3)
            ),

            "ram_available_gb": (
                system_memory.available
                / (1024 ** 3)
            ),

            "ram_percent": (
                system_memory.percent
            ),

            "swap_used_gb": (
                swap_memory.used
                / (1024 ** 3)
            ),

            "swap_percent": (
                swap_memory.percent
            ),
        }

    except Exception as error:
        return {
            "error": str(error)
        }


# ============================================================
# GPU Memory Monitoring
# ============================================================

def get_gpu_memory():
    """
    Read GPU memory usage using nvidia-smi.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",

                "--query-gpu="
                "index,name,memory.used,memory.total",

                "--format=csv,noheader,nounits",
            ],

            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return (
                "nvidia-smi error: "
                + result.stderr.strip()
            )

        return result.stdout.strip()

    except FileNotFoundError:
        return (
            "nvidia-smi was not found"
        )

    except Exception as error:
        return str(error)


# ============================================================
# Memory Monitor Function
# ============================================================

def monitor_memory():
    """
    Continuously monitor CPU RAM and GPU memory.

    This function runs in a background thread.
    """

    log(
        "Memory monitor started",
        memory_log,
    )

    while not stop_monitor_event.is_set():

        # --------------------------------------------
        # RAM Information
        # --------------------------------------------

        memory = get_system_memory()

        if "error" not in memory:

            ram_message = (
                f"RAM used="
                f"{memory['ram_used_gb']:.2f} GB, "

                f"RAM available="
                f"{memory['ram_available_gb']:.2f} GB, "

                f"RAM usage="
                f"{memory['ram_percent']:.1f}%, "

                f"Process RSS="
                f"{memory['process_rss_gb']:.2f} GB, "

                f"Swap used="
                f"{memory['swap_used_gb']:.2f} GB, "

                f"Swap usage="
                f"{memory['swap_percent']:.1f}%"
            )

            log(
                ram_message,
                memory_log,
            )

        else:

            log(
                f"RAM monitor error: "
                f"{memory['error']}",
                memory_log,
            )

        # --------------------------------------------
        # GPU Information
        # --------------------------------------------

        gpu_memory = get_gpu_memory()

        log(
            "GPU memory:\n"
            + gpu_memory,
            memory_log,
        )

        # --------------------------------------------
        # Wait 30 Seconds or Stop Immediately
        # --------------------------------------------

        stop_monitor_event.wait(
            timeout=30
        )

    log(
        "Memory monitor stopped",
        memory_log,
    )


# ============================================================
# Training Command
# ============================================================

def build_training_command():
    """
    Build the MMDetection training command.
    """

    return [
        sys.executable,

        str(
            MMDET_ROOT
            / "tools"
            / "train.py"
        ),

        str(CONFIG_FILE),

        "--work-dir",

        str(WORK_DIR),
    ]


# ============================================================
# Main Training Function
# ============================================================

def main():

    # --------------------------------------------
    # Initial Information
    # --------------------------------------------

    log(
        "Starting RTMDet-Ins training",
        training_log,
    )

    log(
        f"Project root: {PROJECT_ROOT}",
        training_log,
    )

    log(
        f"MMDetection root: {MMDET_ROOT}",
        training_log,
    )

    log(
        f"Config file: {CONFIG_FILE}",
        training_log,
    )

    log(
        f"Training log: {training_log}",
        training_log,
    )

    log(
        f"Memory log: {memory_log}",
        training_log,
    )

    # --------------------------------------------
    # Validate Required Paths
    # --------------------------------------------

    if not PROJECT_ROOT.exists():

        log(
            f"Project root does not exist: "
            f"{PROJECT_ROOT}",
            training_log,
        )

        return

    if not MMDET_ROOT.exists():

        log(
            f"MMDetection root does not exist: "
            f"{MMDET_ROOT}",
            training_log,
        )

        return

    if not CONFIG_FILE.exists():

        log(
            f"Config file does not exist: "
            f"{CONFIG_FILE}",
            training_log,
        )

        return

    # --------------------------------------------
    # Start Memory Monitor Thread
    # --------------------------------------------

    monitor_thread = threading.Thread(
        target=monitor_memory,
        name="MemoryMonitor",
        daemon=True,
    )

    monitor_thread.start()

    process = None

    try:

        # ----------------------------------------
        # Build Training Command
        # ----------------------------------------

        train_command = (
            build_training_command()
        )

        log(
            "Training command:",
            training_log,
        )

        log(
            " ".join(train_command),
            training_log,
        )

        # ----------------------------------------
        # Start MMDetection Training
        # ----------------------------------------

        log(
            "Launching MMDetection training...",
            training_log,
        )

        process = subprocess.Popen(
            train_command,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1,

            cwd=str(MMDET_ROOT),

            env=os.environ.copy(),
        )

        # ----------------------------------------
        # Stream Training Output
        # ----------------------------------------

        with open(
            training_log,
            "a",
            encoding="utf-8",
            buffering=1,
        ) as log_file:

            for line in process.stdout:

                # Print actual MMDetection output
                # directly to the terminal
                print(
                    line,
                    end="",
                    flush=True,
                )

                # Save actual output to log file
                log_file.write(line)
                log_file.flush()

        # ----------------------------------------
        # Wait for Training Process
        # ----------------------------------------

        process.wait()

        # ----------------------------------------
        # Check Training Result
        # ----------------------------------------

        if process.returncode == 0:

            log(
                "Training completed successfully",
                training_log,
            )

        else:

            log(
                "Training failed with return code: "
                f"{process.returncode}",
                training_log,
            )

    except KeyboardInterrupt:

        log(
            "Training interrupted by user",
            training_log,
        )

        if process is not None:
            process.terminate()

    except Exception as error:

        log(
            f"Training launcher error: {error}",
            training_log,
        )

        if process is not None:
            process.terminate()

    finally:

        # ----------------------------------------
        # Stop Memory Monitor
        # ----------------------------------------

        stop_monitor_event.set()

        monitor_thread.join(
            timeout=5
        )

        log(
            "Training launcher finished",
            training_log,
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()