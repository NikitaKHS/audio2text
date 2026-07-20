"""Тесты GPU-диагностики и VRAM-рекомендаций."""

import unittest

from a2t_lib.hardware import GpuInfo, hardware_summary, vram_warning


class TestHardware(unittest.TestCase):
    def test_gpu_summary(self):
        gpu = GpuInfo(0, "RTX Test", memory_total_mb=8192, runtime_ready=True)
        self.assertIn("RTX Test", gpu.summary())
        self.assertIn("8 ГБ", gpu.summary())
        self.assertIn("CUDA готова", gpu.summary())

    def test_float32_large_warns_on_eight_gb(self):
        gpu = GpuInfo(0, "RTX Test", memory_total_mb=8192, runtime_ready=True)
        warning = vram_warning("large-v3", "float32", gpu)
        self.assertIsNotNone(warning)
        self.assertIn("float16", warning or "")

    def test_safe_profile_has_no_warning_on_eight_gb(self):
        gpu = GpuInfo(0, "RTX Test", memory_total_mb=8192, runtime_ready=True)
        self.assertIsNone(vram_warning("large-v3", "float16", gpu))
        self.assertIn("RTX Test", hardware_summary([gpu]))


if __name__ == "__main__":
    unittest.main()
