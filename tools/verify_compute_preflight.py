"""Validate an independently owner-run Kaggle CPU/GPU receipt without account access.

The receipt is a self-report, not proof of GPU entitlement or successful model inference.
No secrets, network or Kaggle API access. No claim of pooled VRAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_CPU_SHA = hashlib.sha256(b"two-worlds-cpu-check" * 16384).hexdigest()
MAX_RECEIPT_BYTES = 32768
SAFE_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,100}$")


def _gib(value, *, allow_null=False):
    if value is None and allow_null:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2048:
        raise ValueError("Invalid capacity in compute receipt")


def validate_receipt(data: dict) -> dict:
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Invalid compute receipt schema")
    if data.get("source") != "owner_operated_kaggle_notebook":
        raise ValueError("Expected account-owner-operated notebook output")
    if any(data.get(k) is not False for k in
           ("model_inference_tested", "weights_downloaded", "credentials_used", "video_generated")):
        raise ValueError("Receipt must not claim model inference, credentials or media generation")
    cpu, gpu = data.get("cpu"), data.get("gpu")
    if not isinstance(cpu, dict) or not isinstance(gpu, dict):
        raise ValueError("Missing CPU or GPU metrics")
    if cpu.get("smoke") != "PASS" or cpu.get("smoke_sha256") != EXPECTED_CPU_SHA:
        raise ValueError("CPU smoke check failed")
    count = cpu.get("logical_cores")
    if type(count) is not int or not 1 <= count <= 1024:
        raise ValueError("Invalid CPU count")
    _gib(cpu.get("ram_gib"), allow_null=True)
    _gib(cpu.get("disk_free_gib"))
    if type(gpu.get("torch_installed")) is not bool or type(gpu.get("cuda_available")) is not bool:
        raise ValueError("Invalid GPU availability flags")
    gpu_count = gpu.get("device_count")
    devices = gpu.get("devices")
    if type(gpu_count) is not int or not 0 <= gpu_count <= 8 or not isinstance(devices, list):
        raise ValueError("Invalid GPU device inventory")
    if len(devices) != gpu_count:
        raise ValueError("GPU device count inconsistent with inventory")
    if gpu_count > 0 and not gpu["cuda_available"]:
        raise ValueError("GPU devices reported without CUDA")
    for device in devices:
        if not isinstance(device, dict) or not isinstance(device.get("name"), str):
            raise ValueError("Invalid GPU inventory item")
        if not SAFE_NAME.fullmatch(device["name"]):
            raise ValueError("Invalid GPU name")
        _gib(device.get("vram_gib"))
    smoke = gpu.get("smoke")
    if smoke not in ("PASS", "FAIL", "SKIPPED"):
        raise ValueError("Invalid GPU smoke status")
    if smoke == "PASS" and not (gpu_count > 0 and gpu["cuda_available"]):
        raise ValueError("False GPU smoke success")
    return {
        "status": "CPU_AND_GPU_CHECK_PASSED" if smoke == "PASS" else "CPU_CHECK_PASSED_GPU_NOT_READY",
        "cpu_logical_cores": count,
        "cpu_ram_gib": cpu.get("ram_gib"),
        "gpu_device_count": gpu_count,
        "vram_gib_per_device": [d["vram_gib"] for d in devices],
        "cuda_smoke": smoke,
        "model_inference_verified": False,
        "gpu_hours_or_availability_verified": False,
        "accounts_or_api_access_verified": False,
        "pooled_vram": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read owner-run hardware receipt (no secrets/API)")
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.receipt.stat().st_size > MAX_RECEIPT_BYTES:
            raise ValueError("Receipt too large")
        data = json.loads(args.receipt.read_text(encoding="utf-8"))
        report = validate_receipt(data)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        parser.exit(2, "BLOCKED: invalid/unverified CPU/GPU receipt\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
