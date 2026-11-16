"""Read ONLY local Kaggle CUDA resources, never allocate a neural model.

This is executed inside each separately authorized Kaggle session; never
adds the VRAM of different T4 cards/accounts into one fictional 30 GiB GPU.
No credentials, usernames, actor media, Kaggle APIs or external network.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

GIB = 1024 ** 3


def hardware_report(torch_module=None, disk_root: Path | str = "/kaggle/working") -> dict:
    base = {
        "schema": 1,
        "local_gpu_measurement_only": True,
        "cuda_available": False,
        "gpu_device_count": 0,
        "per_device": [],
        "vrams_are_pooled": False,
        "total_gpu_memory_valid_for_one_model": False,
        "dual_process_capacity_verified": False,
        "native_576x1024_model_peak_measured": False,
        "remaining_kaggle_weekly_quota_measured": False,
        "neural_model_executed": False,
        "private_media_accessed": False,
        "gpu_jobs_submitted": 0,
        "published": False,
    }
    if torch_module is None:
        try:
            import torch
            torch_module = torch
        except ImportError:
            return {**base, "status": "TORCH_NOT_INSTALLED"}
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {**base, "status": "CUDA_NOT_ALLOCATED"}
    count = cuda.device_count()
    if type(count) is not int or count < 1 or count > 16:
        return {**base, "status": "INVALID_CUDA_DEVICE_COUNT"}
    devices = []
    try:
        for index in range(count):
            prop = cuda.get_device_properties(index)
            installed = int(prop.total_memory)
            free, total_visible = cuda.mem_get_info(index)
            free, total_visible = int(free), int(total_visible)
            if (installed <= 0 or total_visible <= 0 or
                    free < 0 or free > total_visible):
                return {**base, "status": "INVALID_CUDA_FREE_MEMORY_READING"}
            devices.append({
                "index": index,
                "hardware_family": ("NVIDIA_T4"
                                    if "T4" in str(prop.name).upper()
                                    else "OTHER_GPU"),
                "device_vram_gib": round(installed / GIB, 3),
                "currently_free_vram_gib": round(free / GIB, 3),
                "per_device_memory_only": True,
            })
    except (AttributeError, ValueError, TypeError, RuntimeError):
        return {**base, "status": "CUDA_RESOURCE_READ_FAILED"}
    disk = {}
    location = Path(disk_root)
    if location.is_dir():
        try:
            usage = shutil.disk_usage(location)
            disk = {
                "available_disk_gib": round(usage.free / GIB, 3),
                "total_disk_gib": round(usage.total / GIB, 3),
            }
        except OSError:
            pass
    return {
        **base,
        "status": "CUDA_DEVICES_MEASURED_NO_NATIVE_MODEL_PROFILE",
        "cuda_available": True,
        "gpu_device_count": count,
        "per_device": devices,
        "disk_capacity": disk,
        "max_parallel_unreviewed_neural_processes": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/EB01_GPU_HARDWARE_ONLY.json"))
    args = parser.parse_args()
    report = hardware_report()
    p = args.output
    if not p.is_absolute():
        raise ValueError("Hardware audit output must be absolute and private")
    workdir = Path(os.environ.get("GITHUB_WORKSPACE", "/__none__")).resolve()
    if p.resolve().is_relative_to(workdir):
        raise ValueError("Do not commit Kaggle runner reports to public repository")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                 encoding="utf-8")
    if os.name == "posix":
        os.chmod(p, 0o600)
    print("EB01_LOCAL_GPU_HARDWARE_STATUS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
