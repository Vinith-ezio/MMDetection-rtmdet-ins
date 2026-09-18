import os
import gc

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MemoryCleanupHook(Hook):

    priority = "LOW"

    def __init__(
        self,
        interval=1,
        collect_cpu=True,
        clean_cuda=True,
        log_swap=True,
    ):
        self.interval = interval
        self.collect_cpu = collect_cpu
        self.clean_cuda = clean_cuda
        self.log_swap = log_swap

    @staticmethod
    def get_memory_info():

        try:
            import psutil

            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()

            system_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()

            return {
                "process_rss_gb": process_memory.rss / (1024 ** 3),
                "system_used_gb": system_memory.used / (1024 ** 3),
                "system_available_gb": system_memory.available / (1024 ** 3),
                "system_percent": system_memory.percent,
                "swap_used_gb": swap_memory.used / (1024 ** 3),
                "swap_percent": swap_memory.percent,
            }

        except Exception as error:
            return {"error": str(error)}

    @staticmethod
    def cleanup_memory(
        collect_cpu=True,
        clean_cuda=True,
    ):

        collected_objects = 0

        if collect_cpu:
            collected_objects = gc.collect()

        if clean_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()

            except Exception as error:
                print(
                    f"[MemoryCleanupHook] CUDA cleanup warning: {error}"
                )

        return collected_objects

    def before_train(self, runner):
        memory = self.get_memory_info()

        if "error" in memory:
            print(
                f"[MemoryCleanupHook] Initial memory log failed: "
                f"{memory['error']}"
            )
            return

        print(
            "\n"
            "[MemoryCleanupHook] Training started\n"
            f"  Process RSS      : {memory['process_rss_gb']:.2f} GB\n"
            f"  System used      : {memory['system_used_gb']:.2f} GB\n"
            f"  System available : {memory['system_available_gb']:.2f} GB\n"
            f"  System RAM usage : {memory['system_percent']:.1f}%\n"
            f"  Swap used        : {memory['swap_used_gb']:.2f} GB\n"
        )

    def after_train_epoch(self, runner):
        if self.interval <= 0:
            return

        if (runner.epoch + 1) % self.interval != 0:
            return

        before = self.get_memory_info()

        collected = self.cleanup_memory(
            collect_cpu=self.collect_cpu,
            clean_cuda=self.clean_cuda,
        )

        after = self.get_memory_info()

        if "error" in before or "error" in after:
            print(
                "[MemoryCleanupHook] Memory logging failed."
            )
            return

        print(
            "\n"
            "[MemoryCleanupHook] Epoch memory cleanup\n"
            f"  Epoch            : {runner.epoch + 1}\n"
            f"  Process RSS      : "
            f"{before['process_rss_gb']:.2f} -> "
            f"{after['process_rss_gb']:.2f} GB\n"
            f"  System available : "
            f"{before['system_available_gb']:.2f} -> "
            f"{after['system_available_gb']:.2f} GB\n"
            f"  RAM usage        : "
            f"{before['system_percent']:.1f}% -> "
            f"{after['system_percent']:.1f}%\n"
            f"  Python objects   : {collected}\n"
            f"  Swap used        : "
            f"{after['swap_used_gb']:.2f} GB\n"
        )
