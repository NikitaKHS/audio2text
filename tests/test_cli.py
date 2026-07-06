"""Тесты CLI."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcribe_cli import build_parser, config_from_args


class TestCLI(unittest.TestCase):
    def test_list_presets_exits_zero(self):
        with patch("transcribe_cli.main") as mock_main:
            from transcribe_cli import main
        # direct test via parser
        parser = build_parser()
        args = parser.parse_args(["--list-presets"])
        self.assertTrue(args.list_presets)

    def test_config_from_args_basic(self):
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--preset", "balanced", "-o", "out"])
        cfg = config_from_args(args)
        self.assertEqual(cfg.audio_path, Path("test.m4a"))
        self.assertEqual(cfg.output_dir, Path("out"))
        self.assertEqual(cfg.preset, "balanced")

    def test_config_from_args_overrides(self):
        parser = build_parser()
        args = parser.parse_args([
            "a.wav", "--beam-size", "7", "--no-condition-on-previous-text",
        ])
        cfg = config_from_args(args)
        self.assertEqual(cfg.params.beam_size, 7)
        self.assertFalse(cfg.params.condition_on_previous_text)

    def test_config_from_json_file(self):
        data = {
            "audio_path": "rec.m4a",
            "output_dir": "results",
            "preset": "safe",
            "model": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "language": "ru",
            "postprocess": True,
            "params": {"beam_size": 5, "best_of": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "job.json"
            cfg_path.write_text(json.dumps(data), encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["--load-config", str(cfg_path)])
            cfg = config_from_args(args)
            self.assertEqual(cfg.audio_path, Path("rec.m4a"))
            self.assertEqual(cfg.output_dir, Path("results"))

    def test_missing_audio_without_config_exits(self):
        parser = build_parser()
        args = parser.parse_args([])
        with self.assertRaises(SystemExit):
            config_from_args(args)


if __name__ == "__main__":
    unittest.main()
