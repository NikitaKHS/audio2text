"""Тесты форматирования таймкодов."""

import unittest

from a2t_lib.timestamps import format_ts, ts_srt


class TestTimestamps(unittest.TestCase):
    def test_format_ts_zero(self):
        self.assertEqual(format_ts(0), "00:00:00.000")

    def test_format_ts_with_hours(self):
        self.assertEqual(format_ts(3661.5), "01:01:01.500")

    def test_ts_srt_milliseconds(self):
        self.assertEqual(ts_srt(1.234), "00:00:01,234")

    def test_ts_srt_hour(self):
        self.assertEqual(ts_srt(3600), "01:00:00,000")


if __name__ == "__main__":
    unittest.main()
