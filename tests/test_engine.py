"""Тесты движка транскрибации (с моком модели)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from a2t_lib.config import TranscribeConfig
from a2t_lib.engine import _build_transcribe_kwargs, _resolve_device, transcribe_file


class FakeWhisperSegment:
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        avg_logprob: float = -0.3,
        no_speech_prob: float = 0.1,
    ):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class FakeInfo:
    duration = 12.0
    language = "ru"
    language_probability = 0.98


class TestEngine(unittest.TestCase):
    def test_build_transcribe_kwargs_clip(self):
        cfg = TranscribeConfig.from_preset(Path("a.m4a"), preset="safe")
        cfg.clip_start = 100.0
        cfg.clip_end = 200.0
        kw = _build_transcribe_kwargs(cfg)
        self.assertEqual(kw["clip_timestamps"], [100.0, 200.0])

    def test_build_transcribe_kwargs_vad(self):
        cfg = TranscribeConfig.from_preset(Path("a.m4a"), preset="safe")
        kw = _build_transcribe_kwargs(cfg)
        self.assertTrue(kw["vad_filter"])
        self.assertIn("vad_parameters", kw)

    def test_resolve_device_explicit(self):
        self.assertEqual(_resolve_device("cpu"), "cpu")
        self.assertEqual(_resolve_device("cuda"), "cuda")

    def test_transcribe_file_with_mock_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "test.wav"
            audio.write_bytes(b"RIFF fake audio")

            model = MagicMock()
            model.transcribe.return_value = (
                [
                    FakeWhisperSegment(0, 2, "Привет, как дела?"),
                    FakeWhisperSegment(2, 5, "Нормально."),
                    # повторы — должны отфильтроваться на лету
                    FakeWhisperSegment(5, 6, "Спасибо."),
                    FakeWhisperSegment(6, 7, "Спасибо."),
                    FakeWhisperSegment(7, 8, "Спасибо."),
                ],
                FakeInfo(),
            )

            cfg = TranscribeConfig.from_preset(audio, preset="safe")
            logs: list[str] = []
            progress_calls: list[tuple[float, str]] = []

            paths = transcribe_file(
                cfg,
                model=model,
                log=logs.append,
                progress=lambda s, sn: progress_calls.append((s, sn)),
            )

            self.assertTrue(paths["final_txt"].is_file())
            self.assertTrue(paths["final_srt"].is_file())
            self.assertTrue(paths["final_plain"].is_file())
            self.assertTrue(paths["raw_txt"].is_file())
            self.assertTrue(paths["log_txt"].is_file())

            final = paths["final_txt"].read_text(encoding="utf-8")
            self.assertIn("Привет", final)
            self.assertIn("Нормально", final)

            # 3 повтора «Спасибо» — последние 2 отсечены на лету
            raw = paths["raw_txt"].read_text(encoding="utf-8")
            self.assertEqual(raw.count("Спасибо"), 1)

            model.transcribe.assert_called_once()
            call_kwargs = model.transcribe.call_args[1]
            self.assertFalse(call_kwargs["condition_on_previous_text"])

            # progress: init + segments + done
            self.assertTrue(any(p[0] < 0 for p in progress_calls))

    def test_transcribe_file_missing_audio(self):
        cfg = TranscribeConfig.from_preset(Path("nonexistent.wav"), preset="safe")
        with self.assertRaises(FileNotFoundError):
            transcribe_file(cfg, model=MagicMock())

    def test_cancel_raises_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "test.wav"
            audio.write_bytes(b"x")

            def infinite_segments():
                for i in range(100):
                    yield FakeWhisperSegment(i, i + 1, f"seg {i}")

            model = MagicMock()
            model.transcribe.return_value = (infinite_segments(), FakeInfo())

            cfg = TranscribeConfig.from_preset(audio, preset="safe")
            cancelled = False

            def cancel_check():
                nonlocal cancelled
                if not cancelled:
                    cancelled = True
                    return False
                return True

            with self.assertRaises(InterruptedError):
                transcribe_file(cfg, model=model, cancel_check=cancel_check)


if __name__ == "__main__":
    unittest.main()
