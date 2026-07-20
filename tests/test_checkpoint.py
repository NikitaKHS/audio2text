"""Тесты контрольных точек длинной транскрибации."""

import tempfile
import unittest
from pathlib import Path

from a2t_lib.checkpoint import (
    CheckpointMismatchError,
    TranscriptionCheckpoint,
    checkpoint_metadata,
)
from a2t_lib.config import TranscribeConfig
from a2t_lib.postprocess import Segment


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            audio.write_bytes(b"audio")
            cfg = TranscribeConfig.from_preset(audio)
            path = Path(tmp) / "job.partial.jsonl"
            metadata = checkpoint_metadata(cfg, audio.resolve())

            first = TranscriptionCheckpoint(path, metadata, resume=True)
            first.append(Segment(0, 4.5, "Первый сегмент", -0.2, 0.1))
            first.close()

            resumed = TranscriptionCheckpoint(path, metadata, resume=True)
            self.assertEqual(resumed.resume_at, 4.5)
            self.assertEqual(resumed.segments[0].text, "Первый сегмент")
            resumed.complete()
            self.assertFalse(path.exists())

    def test_mismatch_does_not_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            audio.write_bytes(b"audio")
            cfg = TranscribeConfig.from_preset(audio)
            path = Path(tmp) / "job.partial.jsonl"
            original = TranscriptionCheckpoint(
                path, checkpoint_metadata(cfg, audio.resolve()), resume=True
            )
            original.close()

            changed = TranscribeConfig.from_preset(audio)
            changed.params.beam_size = 7
            with self.assertRaises(CheckpointMismatchError):
                TranscriptionCheckpoint(
                    path, checkpoint_metadata(changed, audio.resolve()), resume=True
                )
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
