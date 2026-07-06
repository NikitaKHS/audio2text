"""Загрузка модели, transcribe(), запись результатов на диск."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel
from tqdm import tqdm

from a2t_lib.config import TranscribeConfig
from a2t_lib.postprocess import (
    Segment,
    clean_segments,
    is_live_spam,
    norm,
    segments_to_plain,
    segments_to_srt,
    segments_to_txt,
)
from a2t_lib.timestamps import format_ts

LogFn = Callable[[str], None]
# progress(sec, snippet): sec < 0 значит «total duration = -sec»
ProgressFn = Callable[[float, str], None]


def _log(fn: LogFn | None, msg: str) -> None:
    if fn:
        fn(msg)
    else:
        print(msg, flush=True)


def _find_ffmpeg() -> str | None:
    """faster-whisper декодирует через ffmpeg; без него m4a/mp3 часто не открываются."""
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def load_model(config: TranscribeConfig, log: LogFn | None = None) -> WhisperModel:
    device = _resolve_device(config.device)
    compute = config.effective_compute_type()
    if device == "cpu" and compute in ("float16", "bfloat16"):
        compute = "float32"
        _log(log, "CPU: compute_type переключён на float32")

    _log(log, f"Загрузка {config.model} ({compute}) на {device}...")
    return WhisperModel(config.model, device=device, compute_type=compute)


def _build_transcribe_kwargs(config: TranscribeConfig) -> dict:
    p = config.params
    kwargs: dict = {
        "language": config.language,
        "beam_size": p.beam_size,
        "best_of": p.best_of,
        "patience": p.patience,
        "temperature": p.temperature,
        "vad_filter": p.vad_filter,
        "condition_on_previous_text": p.condition_on_previous_text,
        "no_speech_threshold": p.no_speech_threshold,
        "compression_ratio_threshold": p.compression_ratio_threshold,
        "hallucination_silence_threshold": p.hallucination_silence_threshold,
        "initial_prompt": config.initial_prompt or None,
        "hotwords": config.hotwords or None,
    }
    if p.vad_filter:
        kwargs["vad_parameters"] = {
            "threshold": p.vad_threshold,
            "min_speech_duration_ms": p.vad_min_speech_ms,
            "min_silence_duration_ms": p.vad_min_silence_ms,
        }
    if p.word_timestamps:
        kwargs["word_timestamps"] = True
    if config.clip_start is not None:
        # список [start] или [start, end] — не строка "123," (ломает faster-whisper)
        clip = [config.clip_start]
        if config.clip_end is not None:
            clip.append(config.clip_end)
        kwargs["clip_timestamps"] = clip
    return kwargs


def transcribe_file(
    config: TranscribeConfig,
    model: WhisperModel | None = None,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """
    Один файл от начала до конца.

    model можно передать снаружи (тесты). log/progress — для GUI.
    cancel_check вызывается на каждом сегменте; бросает InterruptedError.
    """
    audio = config.audio_path.resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio}")

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        _log(log, f"ffmpeg: {ffmpeg}")
    else:
        _log(log, "Предупреждение: ffmpeg не найден в PATH — возможны ошибки декодирования")

    out_dir = config.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = config.output_paths()

    if model is None:
        model = load_model(config, log)

    p = config.params
    _log(
        log,
        f"\n=== {audio.name} ===\n"
        f"Пресет: {config.preset} | модель: {config.model} | "
        f"beam={p.beam_size} | condition_prev={p.condition_on_previous_text}",
    )

    kwargs = _build_transcribe_kwargs(config)
    segments_iter, info = model.transcribe(str(audio), **kwargs)
    duration = info.duration or 0
    _log(
        log,
        f"Длительность: {format_ts(duration)} | "
        f"язык: {info.language} ({info.language_probability:.0%})",
    )
    if progress and duration:
        progress(-duration, "")

    raw_segments: list[Segment] = []
    review_lines: list[str] = []
    uncertain = 0
    live_dropped = 0
    last_norm = ""
    repeat_streak = 0

    # CLI без callbacks — tqdm; GUI передаёт progress
    use_tqdm = log is None and progress is None
    bar = None
    if use_tqdm:
        bar = tqdm(
            total=int(duration) or None,
            unit="s",
            desc="Транскрибация",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        )

    for seg in segments_iter:
        if cancel_check and cancel_check():
            _log(log, "Отменено пользователем.")
            if bar:
                bar.close()
            raise InterruptedError("Транскрибация отменена")

        text = seg.text.strip()
        n = norm(text)
        if n == last_norm and n:
            repeat_streak += 1
        else:
            repeat_streak = 1 if n else 0
            last_norm = n

        # отсечь серию одинаковых «спасибо» / однословных повторов до записи в raw
        if is_live_spam(text, repeat_streak):
            live_dropped += 1
            if bar:
                bar.update(max(0, int(seg.end) - bar.n))
            continue

        low_conf = seg.avg_logprob < p.low_confidence_logprob or seg.no_speech_prob > 0.6
        if low_conf:
            uncertain += 1
            review_lines.append(
                f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] "
                f"logprob={seg.avg_logprob:.2f}, no_speech={seg.no_speech_prob:.2f}\n{text}\n"
            )

        raw_segments.append(
            Segment(seg.start, seg.end, text, seg.avg_logprob, seg.no_speech_prob)
        )

        snippet = text[:60] + ("…" if len(text) > 60 else "")
        if bar:
            bar.set_postfix_str(snippet)
            bar.update(max(0, int(seg.end) - bar.n))
        elif progress:
            progress(seg.end, snippet)

    if bar:
        bar.close()
    elif progress and duration:
        progress(duration, "Готово")

    if live_dropped:
        _log(log, f"Отфильтровано на лету (повторы): {live_dropped} сегментов")

    if config.save_raw:
        raw_header = (
            f"# Raw transcript | {audio.name} | preset={config.preset} | model={config.model}\n"
        )
        paths["raw_txt"].write_text(
            segments_to_txt(raw_segments, raw_header), encoding="utf-8"
        )
        _log(log, f"Сохранено: {paths['raw_txt'].name} ({len(raw_segments)} сегментов)")

    if config.postprocess:
        final_segments, clean_log = clean_segments(raw_segments, audio.name)
        paths["log_txt"].write_text("\n".join(clean_log), encoding="utf-8")
        _log(
            log,
            f"Постобработка: {len(raw_segments)} -> {len(final_segments)} сегментов",
        )
    else:
        final_segments = raw_segments

    header = f"# Транскрипт | {audio.name} | preset={config.preset}\n"
    paths["final_txt"].write_text(
        segments_to_txt(final_segments, header), encoding="utf-8"
    )
    _log(log, f"Сохранено: {paths['final_txt'].name}")

    if config.save_srt:
        paths["final_srt"].write_text(segments_to_srt(final_segments), encoding="utf-8")
        _log(log, f"Сохранено: {paths['final_srt'].name}")

    if config.save_plain:
        paths["final_plain"].write_text(segments_to_plain(final_segments), encoding="utf-8")
        _log(log, f"Сохранено: {paths['final_plain'].name}")

    if config.save_review and review_lines:
        paths["review_txt"].write_text("\n".join(review_lines), encoding="utf-8")
        _log(log, f"Сохранено: {paths['review_txt'].name} ({uncertain} сомнительных)")

    _log(log, "\nГотово.")
    return paths
