"""Measure the actual model process, per-card VRAM, CPU/RAM/disk, no prompts."""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path


def start(path: str):
    import psutil
    import torch
    stop = threading.Event()
    started = time.monotonic()
    process = psutil.Process()
    report = {"schema": 1, "samples": 0, "per_device": [],
              "host_ram_peak_gib": 0., "process_rss_peak_gib": 0.,
              "cpu_percent_peak": 0., "cpu_logical_count": psutil.cpu_count(),
              "vrams_are_pooled": False, "finished": False}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        torch.cuda.reset_peak_memory_stats(i)
        report["per_device"].append({"index": i, "name": p.name,
            "physical_gib": round(p.total_memory / 2**30, 4),
            "torch_max_allocated_gib": 0., "torch_max_reserved_gib": 0.,
            "device_memory_used_peak_gib": 0.})
    lock = threading.Lock()
    def sample():
        with lock:
            report["samples"] += 1
            memory = psutil.virtual_memory()
            report["host_ram_total_gib"] = round(memory.total / 2**30, 4)
            report["host_ram_peak_gib"] = max(report["host_ram_peak_gib"], round(memory.used / 2**30, 4))
            report["host_ram_available_gib"] = round(memory.available / 2**30, 4)
            report["process_rss_peak_gib"] = max(report["process_rss_peak_gib"], round(process.memory_info().rss / 2**30, 4))
            report["cpu_percent_peak"] = max(report["cpu_percent_peak"], process.cpu_percent())
            report["disk_free_gib"] = round(psutil.disk_usage('/kaggle/working').free / 2**30, 4)
            report["wall_seconds"] = round(time.monotonic() - started, 3)
            for gpu in report["per_device"]:
                i = gpu["index"]
                free, total = torch.cuda.mem_get_info(i)
                gpu["device_memory_used_peak_gib"] = max(gpu["device_memory_used_peak_gib"], round((total-free) / 2**30, 4))
                gpu["torch_max_allocated_gib"] = round(torch.cuda.max_memory_allocated(i) / 2**30, 4)
                gpu["torch_max_reserved_gib"] = round(torch.cuda.max_memory_reserved(i) / 2**30, 4)
            dest = Path(path)
            temporary = dest.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, sort_keys=True))
            temporary.replace(dest)
    def worker():
        while not stop.wait(2):
            sample()
    sample()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    def finish():
        stop.set()
        thread.join(timeout=4)
        report["finished"] = True
        sample()
        return report
    return finish
