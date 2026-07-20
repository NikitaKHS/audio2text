"""Тесты конфигурации и пресетов."""

import json
import unittest
from pathlib import Path

from a2t_lib.config import (
    DEFAULT_HOTWORDS,
    DEFAULT_PROMPT,
    PRESETS,
    TranscribeConfig,
    TranscribeParams,
)


class TestConfig(unittest.TestCase):
    def test_all_presets_have_required_keys(self):
        for name, meta in PRESETS.items():
            self.assertIn("description", meta, name)
            self.assertIn("model", meta, name)
            self.assertIn("compute_type", meta, name)
            self.assertIn("params", meta, name)
            self.assertIsInstance(meta["params"], TranscribeParams, name)

    def test_from_preset_applies_defaults(self):
        cfg = TranscribeConfig.from_preset(Path("audio.m4a"), preset="safe")
        self.assertEqual(cfg.preset, "safe")
        self.assertEqual(cfg.model, "large-v3")
        self.assertEqual(cfg.compute_type, "float16")
        self.assertFalse(cfg.params.condition_on_previous_text)

    def test_max_quality_preset(self):
        cfg = TranscribeConfig.from_preset(Path("x.wav"), preset="max_quality")
        self.assertEqual(cfg.params.beam_size, 10)
        self.assertEqual(cfg.compute_type, "float32")

    def test_unknown_preset_raises(self):
        cfg = TranscribeConfig(audio_path=Path("a.wav"))
        with self.assertRaises(ValueError):
            cfg.apply_preset("nonexistent")

    def test_output_paths(self):
        cfg = TranscribeConfig(
            audio_path=Path(r"C:\rec\session.m4a"),
            output_dir=Path(r"C:\out"),
            stem="custom",
        )
        paths = cfg.output_paths()
        self.assertEqual(paths["final_txt"], Path(r"C:\out\custom_final.txt"))
        self.assertEqual(paths["raw_txt"], Path(r"C:\out\custom_transcript_raw.txt"))
        self.assertEqual(paths["checkpoint"], Path(r"C:\out\custom_transcript.partial.jsonl"))

    def test_effective_compute_type_default(self):
        cfg = TranscribeConfig.from_preset(Path("a.wav"), preset="fast")
        cfg.compute_type = "default"
        self.assertEqual(cfg.effective_compute_type(), "int8_float16")

    def test_to_dict_roundtrip_fields(self):
        cfg = TranscribeConfig.from_preset(Path("a.wav"), preset="balanced")
        d = cfg.to_dict()
        self.assertEqual(d["preset"], "balanced")
        self.assertEqual(d["params"]["beam_size"], 5)
        self.assertIsInstance(json.dumps(d, ensure_ascii=False), str)

    def test_preset_params_are_not_shared_between_jobs(self):
        first = TranscribeConfig.from_preset(Path("first.wav"), preset="safe")
        second = TranscribeConfig.from_preset(Path("second.wav"), preset="safe")
        first.params.beam_size = 99
        self.assertEqual(second.params.beam_size, 5)
        self.assertEqual(PRESETS["safe"]["params"].beam_size, 5)

    def test_validate_rejects_invalid_clip_range(self):
        cfg = TranscribeConfig.from_preset(Path("a.wav"), preset="safe")
        cfg.clip_start = 10
        cfg.clip_end = 5
        with self.assertRaisesRegex(ValueError, "clip_end"):
            cfg.validate()

    def test_validate_rejects_path_traversal_in_stem(self):
        cfg = TranscribeConfig.from_preset(Path("a.wav"), preset="safe", stem="../outside")
        with self.assertRaisesRegex(ValueError, "Префикс"):
            cfg.validate()

    def test_default_context_is_neutral(self):
        self.assertEqual(DEFAULT_PROMPT, "")
        self.assertEqual(DEFAULT_HOTWORDS, "")

    def test_from_dict_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "Неизвестные поля"):
            TranscribeConfig.from_dict({"audio_path": "a.wav", "surprise": True})


if __name__ == "__main__":
    unittest.main()
