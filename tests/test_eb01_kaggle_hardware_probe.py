"""GPU-free validation of Kaggle CUDA inventory, never pooled VRAM."""
import unittest
from types import SimpleNamespace

from tools.eb01_kaggle_hardware_probe import hardware_report, GIB


class FakeCuda:
    def __init__(self, devices):
        self.devices = devices

    def is_available(self):
        return bool(self.devices)

    def device_count(self):
        return len(self.devices)

    def get_device_properties(self, n):
        return SimpleNamespace(name="Tesla T4",
                               total_memory=self.devices[n][0])

    def mem_get_info(self, n):
        return self.devices[n][1], self.devices[n][0]


class HardwareTests(unittest.TestCase):
    def test_two_t4_devices_are_independent_and_not_30gb_one_model(self):
        torch = SimpleNamespace(cuda=FakeCuda([
            (int(14.56*GIB), int(13.4*GIB)),
            (int(14.56*GIB), int(13.7*GIB)),
        ]))
        r = hardware_report(torch_module=torch, disk_root="/__no_disk__")
        self.assertEqual(r["status"], "CUDA_DEVICES_MEASURED_NO_NATIVE_MODEL_PROFILE")
        self.assertEqual(r["gpu_device_count"], 2)
        self.assertEqual(r["per_device"][0]["hardware_family"], "NVIDIA_T4")
        self.assertEqual(r["max_parallel_unreviewed_neural_processes"], 1)
        self.assertFalse(r["vrams_are_pooled"])
        self.assertFalse(r["dual_process_capacity_verified"])
        self.assertFalse(r["native_576x1024_model_peak_measured"])
        self.assertFalse(r["remaining_kaggle_weekly_quota_measured"])
        self.assertFalse(r["neural_model_executed"])
        self.assertEqual(r["gpu_jobs_submitted"], 0)

    def test_no_cuda_is_never_claimed_as_render_capable(self):
        r = hardware_report(torch_module=SimpleNamespace(cuda=FakeCuda([])))
        self.assertEqual(r["status"], "CUDA_NOT_ALLOCATED")
        self.assertFalse(r["cuda_available"])

    def test_unreliable_or_oversubscribed_cuda_free_memory_is_blocked(self):
        torch = SimpleNamespace(cuda=FakeCuda([(16*GIB, 18*GIB)]))
        r = hardware_report(torch_module=torch, disk_root="/__no_disk__")
        self.assertEqual(r["status"], "INVALID_CUDA_FREE_MEMORY_READING")
        self.assertFalse(r["cuda_available"])


if __name__ == "__main__":
    unittest.main()
