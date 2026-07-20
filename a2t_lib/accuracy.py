"""Высокоточная русская транскрибация с независимой проверкой двумя GigaAM-v3."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from a2t_lib.engine import _write_text
from a2t_lib.postprocess import (
    Segment,
    is_junk,
    segments_to_plain,
    segments_to_srt,
    segments_to_txt,
)
from a2t_lib.timestamps import format_ts

GIGAAM_COMMIT = "559d88d6b72541412743929f633a6ae7c9950b85"
DEFAULT_PRIMARY_MODEL = "v3_e2e_rnnt"
DEFAULT_VERIFY_MODEL = "v3_e2e_ctc"


def accuracy_installed() -> bool:
    """Быстрая проверка optional-режима без тяжёлого импорта PyTorch."""
    from importlib.util import find_spec

    return find_spec("gigaam") is not None and find_spec("torch") is not None


@dataclass(frozen=True)
class SpeechChunk:
    index: int
    start_sample: int
    end_sample: int
    sample_rate: int = 16_000

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass(frozen=True)
class AccuracyRow:
    chunk: SpeechChunk
    primary_text: str
    verify_text: str

    @property
    def agreement(self) -> float:
        return text_agreement(self.primary_text, self.verify_text)


@dataclass(frozen=True)
class ManualCorrection:
    start: float
    end: float
    find: str
    replace: str
    note: str = ""


def _normalise_for_comparison(text: str) -> str:
    text = text.casefold().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9]+", " ", text).strip()


def text_agreement(left: str, right: str) -> float:
    """Сходство двух независимых гипотез от 0 до 1."""
    left_norm = _normalise_for_comparison(left)
    right_norm = _normalise_for_comparison(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def build_speech_chunks(audio: Any, *, sample_rate: int = 16_000) -> list[SpeechChunk]:
    """Делит речь по паузам на куски короче лимита GigaAM (25 секунд)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=180,
        max_speech_duration_s=22.0,
        min_silence_duration_ms=500,
        speech_pad_ms=160,
    )
    timestamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    return [
        SpeechChunk(i, int(item["start"]), int(item["end"]), sample_rate)
        for i, item in enumerate(timestamps)
        if int(item["end"]) > int(item["start"])
    ]


def _transcribe_samples(model: Any, samples: Any) -> str:
    """Быстрый вызов GigaAM без сотен временных WAV-файлов."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - понятная ошибка для optional dependency
        raise RuntimeError(
            "Высокоточный режим не установлен. Запустите setup_accuracy.bat."
        ) from exc

    contiguous = np.ascontiguousarray(samples)
    wav = torch.from_numpy(contiguous).to(model._device).to(model._dtype).unsqueeze(0)
    length = torch.full([1], wav.shape[-1], device=model._device)
    with torch.inference_mode():
        encoded, encoded_len = model.forward(wav, length)
        decoded = model._decode(encoded, encoded_len, length, False)
    return str(decoded[0][0]).strip()


def load_corrections(path: Path | None, audio_name: str) -> list[ManualCorrection]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get(audio_name, []) if isinstance(data, dict) else []
    corrections: list[ManualCorrection] = []
    for row in rows:
        corrections.append(
            ManualCorrection(
                start=float(row["start"]),
                end=float(row["end"]),
                find=str(row["find"]),
                replace=str(row["replace"]),
                note=str(row.get("note", "")),
            )
        )
    return corrections


def apply_manual_corrections(
    segments: list[Segment], corrections: list[ManualCorrection]
) -> tuple[list[Segment], list[str]]:
    log: list[str] = []
    output: list[Segment] = []
    used: set[int] = set()
    for segment in segments:
        text = segment.text
        for i, correction in enumerate(corrections):
            if segment.end <= correction.start or segment.start >= correction.end:
                continue
            if correction.find not in text:
                continue
            text = text.replace(correction.find, correction.replace, 1)
            used.add(i)
            note = f" — {correction.note}" if correction.note else ""
            log.append(
                f"[{format_ts(segment.start)} -> {format_ts(segment.end)}] "
                f"{correction.find!r} → {correction.replace!r}{note}"
            )
        output.append(Segment(segment.start, segment.end, text))
    for i, correction in enumerate(corrections):
        if i not in used:
            log.append(
                f"НЕ ПРИМЕНЕНО [{format_ts(correction.start)} -> "
                f"{format_ts(correction.end)}]: {correction.find!r}"
            )
    return output, log


def _checkpoint_metadata(
    audio_path: Path, primary_model: str, verify_model: str, chunks: list[SpeechChunk]
) -> dict[str, Any]:
    stat = audio_path.stat()
    return {
        "schema": 1,
        "audio": str(audio_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "primary_model": primary_model,
        "verify_model": verify_model,
        "chunks": len(chunks),
    }


def _load_checkpoint(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:] if line.strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if header != metadata:
        return []
    expected = list(range(len(rows)))
    if [int(row.get("index", -1)) for row in rows] != expected:
        return []
    return rows


def _open_checkpoint(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        return path.open("a", encoding="utf-8", newline="\n")
    handle = path.open("w", encoding="utf-8", newline="\n")
    handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def high_accuracy_paths(audio_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    out = (output_dir or audio_path.parent).resolve()
    stem = audio_path.stem
    return {
        "txt": out / f"{stem}_high_accuracy.txt",
        "plain": out / f"{stem}_high_accuracy_plain.txt",
        "srt": out / f"{stem}_high_accuracy.srt",
        "review": out / f"{stem}_high_accuracy_review.txt",
        "log": out / f"{stem}_high_accuracy_corrections.txt",
        "checkpoint": out / f"{stem}_high_accuracy.partial.jsonl",
    }


def transcribe_high_accuracy(
    audio_path: Path,
    *,
    output_dir: Path | None = None,
    corrections_path: Path | None = None,
    primary_model_name: str = DEFAULT_PRIMARY_MODEL,
    verify_model_name: str = DEFAULT_VERIFY_MODEL,
    agreement_threshold: float = 0.86,
    progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Создаёт локальный RU-транскрипт и проверяет каждый фрагмент вторым декодером."""
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    try:
        import gigaam
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise RuntimeError(
            "Высокоточный режим не установлен. Запустите setup_accuracy.bat."
        ) from exc

    paths = high_accuracy_paths(audio_path, output_dir)
    paths["txt"].parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"Декодирование {audio_path.name} и поиск речевых фрагментов…")
    audio = decode_audio(str(audio_path), sampling_rate=16_000)
    chunks = build_speech_chunks(audio)
    if not chunks:
        raise RuntimeError(f"В файле {audio_path.name} речь не найдена")

    metadata = _checkpoint_metadata(audio_path, primary_model_name, verify_model_name, chunks)
    saved_rows = _load_checkpoint(paths["checkpoint"], metadata)
    if saved_rows and log:
        log(f"Найдена контрольная точка: {len(saved_rows)}/{len(chunks)} фрагментов")
    if cancel_check and cancel_check():
        raise InterruptedError("Высокоточная транскрибация отменена")
    if log:
        log(f"Загрузка {primary_model_name}…")
    primary_model = gigaam.load_model(primary_model_name)
    if cancel_check and cancel_check():
        raise InterruptedError("Высокоточная транскрибация отменена")
    if log:
        log(f"Загрузка проверочного {verify_model_name}…")
    verify_model = gigaam.load_model(verify_model_name)
    if log:
        log(f"Двойная проверка: {len(chunks)} речевых фрагментов")
    with _open_checkpoint(paths["checkpoint"], metadata, saved_rows) as checkpoint:
        for chunk in chunks[len(saved_rows) :]:
            if cancel_check and cancel_check():
                raise InterruptedError("Высокоточная транскрибация отменена")
            samples = audio[chunk.start_sample : chunk.end_sample]
            primary_text = _transcribe_samples(primary_model, samples)
            if cancel_check and cancel_check():
                raise InterruptedError("Высокоточная транскрибация отменена")
            verify_text = _transcribe_samples(verify_model, samples)
            row = {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "primary": primary_text,
                "verify": verify_text,
            }
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            saved_rows.append(row)
            if progress:
                progress(len(saved_rows), len(chunks), chunk.end)

    rows = [
        AccuracyRow(
            SpeechChunk(
                int(row["index"]),
                round(float(row["start"]) * 16_000),
                round(float(row["end"]) * 16_000),
            ),
            str(row["primary"]),
            str(row["verify"]),
        )
        for row in saved_rows
    ]
    segments: list[Segment] = []
    discarded: list[str] = []
    review: list[str] = [
        f"# Контроль точности — {audio_path.name}",
        f"# Основной: {primary_model_name}; проверочный: {verify_model_name}",
        "# Низкое согласие моделей не означает ошибку, но требует прослушивания.\n",
    ]
    for row in rows:
        segment = Segment(row.chunk.start, row.chunk.end, row.primary_text)
        junk, reason = is_junk(segment)
        if junk:
            discarded.append(
                f"[{format_ts(segment.start)} -> {format_ts(segment.end)}] "
                f"Удалено ({reason}): {segment.text}"
            )
            continue
        segments.append(segment)
        if row.agreement < agreement_threshold:
            review.extend(
                [
                    f"[{format_ts(row.chunk.start)} -> {format_ts(row.chunk.end)}] "
                    f"согласие={row.agreement:.0%}",
                    f"RNNT: {row.primary_text}",
                    f"CTC:  {row.verify_text}\n",
                ]
            )

    corrections = load_corrections(corrections_path, audio_path.name)
    segments, correction_log = apply_manual_corrections(segments, corrections)
    header = (
        f"# Высокоточный транскрипт | {audio_path.name} | "
        f"{primary_model_name} + {verify_model_name}"
    )
    _write_text(paths["txt"], segments_to_txt(segments, header))
    _write_text(paths["plain"], segments_to_plain(segments) + "\n")
    _write_text(paths["srt"], segments_to_srt(segments))
    _write_text(paths["review"], "\n".join(review).rstrip() + "\n")
    log_lines = [
        f"# Ручные исправления — {audio_path.name}",
        *correction_log,
        "",
        "# Автоматически отброшено",
        *(discarded or ["Ничего."]),
    ]
    _write_text(paths["log"], "\n".join(log_lines).rstrip() + "\n")
    paths["checkpoint"].unlink(missing_ok=True)
    if log:
        flagged = sum(row.agreement < agreement_threshold for row in rows)
        log(f"Готово: {len(segments)} сегментов; для проверки отмечено {flagged}")
    return paths
