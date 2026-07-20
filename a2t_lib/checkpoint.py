"""Надёжные контрольные точки для многочасовой транскрибации."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from a2t_lib.config import TranscribeConfig
from a2t_lib.postprocess import Segment

CHECKPOINT_VERSION = 1


class CheckpointMismatchError(ValueError):
    """Контрольная точка относится к другому файлу или набору настроек."""


def checkpoint_metadata(config: TranscribeConfig, audio: Path) -> dict:
    stat = audio.stat()
    return {
        "type": "metadata",
        "version": CHECKPOINT_VERSION,
        "audio": {
            "path": str(audio),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "recognition": {
            "preset": config.preset,
            "model": config.model,
            "device": config.device,
            "device_index": config.device_index,
            "compute_type": config.compute_type,
            "language": config.language,
            "initial_prompt": config.initial_prompt,
            "hotwords": config.hotwords,
            "clip_start": config.clip_start,
            "clip_end": config.clip_end,
            "params": asdict(config.params),
        },
    }


def _atomic_lines(path: Path, records: list[dict]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            for record in records:
                temp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _segment_record(segment: Segment) -> dict:
    return {"type": "segment", **asdict(segment)}


class TranscriptionCheckpoint:
    """Append-only JSONL, который остаётся читаемым после отмены или сбоя."""

    def __init__(self, path: Path, metadata: dict, resume: bool) -> None:
        self.path = path
        self.metadata = metadata
        self.segments: list[Segment] = []
        self._file: TextIO | None = None
        self._pending_fsync = 0
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.is_file() and resume:
            self.segments = self._load_existing()

        # Перезапись также удаляет оборванную последнюю JSON-строку после аварии.
        records = [metadata, *(_segment_record(segment) for segment in self.segments)]
        try:
            _atomic_lines(path, records)
        except PermissionError as exc:
            raise RuntimeError(
                "Эта транскрибация уже запущена в другом процессе. "
                "Закройте второй экземпляр приложения и повторите попытку."
            ) from exc
        self._file = path.open("a", encoding="utf-8", newline="\n", buffering=1)

    @property
    def resume_at(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def _load_existing(self) -> list[Segment]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise CheckpointMismatchError(f"Пустая контрольная точка: {self.path}")
        try:
            existing_metadata = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise CheckpointMismatchError(f"Повреждена контрольная точка: {self.path}") from exc
        if existing_metadata != self.metadata:
            raise CheckpointMismatchError(
                "Контрольная точка создана для другого аудио или настроек. "
                "Верните прежние настройки либо отключите «Продолжить незавершённый запуск»."
            )

        segments: list[Segment] = []
        for index, line in enumerate(lines[1:], start=2):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines):
                    break
                raise CheckpointMismatchError(
                    f"Повреждена строка {index} контрольной точки: {self.path}"
                ) from None
            if record.get("type") != "segment":
                continue
            try:
                segments.append(
                    Segment(
                        start=float(record["start"]),
                        end=float(record["end"]),
                        text=str(record["text"]),
                        avg_logprob=float(record.get("avg_logprob", 0.0)),
                        no_speech_prob=float(record.get("no_speech_prob", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CheckpointMismatchError(
                    f"Некорректный сегмент в строке {index}: {self.path}"
                ) from exc
        return segments

    def append(self, segment: Segment) -> None:
        if self._file is None:
            raise RuntimeError("Контрольная точка уже закрыта")
        self.segments.append(segment)
        self._file.write(
            json.dumps(_segment_record(segment), ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._file.flush()
        self._pending_fsync += 1
        if self._pending_fsync >= 10:
            os.fsync(self._file.fileno())
            self._pending_fsync = 0

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

    def complete(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)
