"""Загрузка модели, transcribe(), запись результатов на диск."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from a2t_lib.checkpoint import TranscriptionCheckpoint, checkpoint_metadata
from a2t_lib.config import TranscribeConfig
from a2t_lib.hardware import configure_cuda_runtime
from a2t_lib.postprocess import (
    Segment,
    clean_segments,
    is_live_spam,
    norm,
    segments_to_plain,
    segments_to_srt,
    segments_to_txt,
)
from a2t_lib.runtime import prevent_system_sleep
from a2t_lib.timestamps import format_ts

LogFn = Callable[[str], None]
# progress(sec, snippet): sec < 0 значит «total duration = -sec»
ProgressFn = Callable[[float, str], None]


class TranscribeModel(Protocol):
    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


def _log(fn: LogFn | None, msg: str) -> None:
    if fn:
        fn(msg)
    else:
        print(msg, flush=True)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def load_model(config: TranscribeConfig, log: LogFn | None = None) -> TranscribeModel:
    if sys.platform == "win32":
        # Xet иногда зависает на многогигабайтных моделях в Windows; обычный HTTP стабильнее.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_runtime = configure_cuda_runtime()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен faster-whisper. Выполните: python -m pip install -r requirements.txt"
        ) from exc

    device = _resolve_device(config.device)
    if device == "cuda":
        if not cuda_runtime.ready:
            missing = ", ".join(cuda_runtime.missing)
            raise RuntimeError(f"Не установлены CUDA-библиотеки: {missing}. Запустите setup.bat.")
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception:
            cuda_count = 0
        if config.device_index >= cuda_count:
            raise RuntimeError(
                f"GPU с индексом {config.device_index} недоступен. Найдено GPU: {cuda_count}."
            )
    compute = config.effective_compute_type()
    if device == "cpu" and compute in ("float16", "int8_float16", "bfloat16"):
        compute = "int8"
        _log(log, "CPU: compute_type автоматически переключён на int8")

    device_label = f"cuda:{config.device_index}" if device == "cuda" else device
    _log(log, f"Загрузка {config.model} ({compute}) на {device_label}...")
    try:
        model_kwargs: dict[str, Any] = {"device": device, "compute_type": compute}
        if device == "cuda":
            model_kwargs["device_index"] = config.device_index
        return WhisperModel(config.model, **model_kwargs)
    except Exception as exc:
        hint = (
            " Проверьте установку CUDA 12/cuDNN 9 или выберите --device cpu."
            if device == "cuda"
            else " Проверьте доступ к диску, сети и кэшу моделей."
        )
        message = f"Не удалось загрузить модель {config.model} на {device_label}.{hint}"
        raise RuntimeError(message) from exc


def _write_text(path: Path, content: str) -> None:
    """Атомарно заменяет файл, не оставляя обрезанный результат при сбое."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


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


@prevent_system_sleep
def transcribe_file(
    config: TranscribeConfig,
    model: TranscribeModel | None = None,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """
    Один файл от начала до конца.

    model можно передать снаружи (тесты). log/progress — для GUI.
    cancel_check вызывается на каждом сегменте; бросает InterruptedError.
    """
    config.validate(check_audio=True)
    audio = config.audio_path.expanduser().resolve()
    config.audio_path = audio

    out_dir = config.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = config.output_paths()

    checkpoint = TranscriptionCheckpoint(
        paths["checkpoint"], checkpoint_metadata(config, audio), resume=config.resume
    )
    raw_segments = checkpoint.segments
    resume_at = checkpoint.resume_at
    if resume_at:
        _log(
            log,
            f"Найдена контрольная точка: {len(raw_segments)} сегментов. "
            f"Продолжение с {format_ts(resume_at)}.",
        )

    try:
        if cancel_check and cancel_check():
            raise InterruptedError("Транскрибация отменена")
        if model is None:
            model = load_model(config, log)
        if cancel_check and cancel_check():
            raise InterruptedError("Транскрибация отменена")
    except Exception:
        checkpoint.close()
        raise

    p = config.params
    _log(
        log,
        f"\n=== {audio.name} ===\n"
        f"Пресет: {config.preset} | модель: {config.model} | "
        f"beam={p.beam_size} | condition_prev={p.condition_on_previous_text}",
    )

    kwargs = _build_transcribe_kwargs(config)
    if resume_at:
        original_start = config.clip_start or 0.0
        effective_start = max(original_start, resume_at)
        clip = [effective_start]
        if config.clip_end is not None:
            clip.append(config.clip_end)
        kwargs["clip_timestamps"] = clip
    try:
        segments_iter, info = model.transcribe(str(audio), **kwargs)
    except Exception as exc:
        checkpoint.close()
        raise RuntimeError(f"Не удалось открыть или обработать аудиофайл: {audio.name}") from exc
    duration = float(getattr(info, "duration", 0) or 0)
    language = getattr(info, "language", config.language) or config.language
    language_probability = getattr(info, "language_probability", None)
    probability_text = (
        f" ({float(language_probability):.0%})" if language_probability is not None else ""
    )
    _log(
        log,
        f"Длительность: {format_ts(duration)} | язык: {language}{probability_text}",
    )
    if progress and duration:
        progress(-duration, "")
        if resume_at:
            progress(resume_at, "Продолжение из контрольной точки")

    review_lines: list[str] = []
    uncertain = 0
    live_dropped = 0
    last_norm = norm(raw_segments[-1].text) if raw_segments else ""
    repeat_streak = 0
    for existing in reversed(raw_segments):
        if norm(existing.text) != last_norm:
            break
        repeat_streak += 1
    for existing in raw_segments:
        if existing.avg_logprob < p.low_confidence_logprob or existing.no_speech_prob > 0.6:
            uncertain += 1
            review_lines.append(
                f"[{format_ts(existing.start)} -> {format_ts(existing.end)}] "
                f"logprob={existing.avg_logprob:.2f}, "
                f"no_speech={existing.no_speech_prob:.2f}\n{existing.text}\n"
            )

    # CLI без callbacks — tqdm; GUI передаёт progress
    use_tqdm = log is None and progress is None
    bar = None
    if use_tqdm:
        try:
            from tqdm import tqdm

            bar = tqdm(
                total=int(duration) or None,
                initial=int(resume_at),
                unit="s",
                desc="Транскрибация",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
            )
        except ImportError:
            bar = None

    try:
        for seg in segments_iter:
            if cancel_check and cancel_check():
                _log(log, "Отменено пользователем.")
                raise InterruptedError("Транскрибация отменена")

            text = seg.text.strip()
            n = norm(text)
            if n == last_norm and n:
                repeat_streak += 1
            else:
                repeat_streak = 1 if n else 0
                last_norm = n

            snippet = text[:60] + ("…" if len(text) > 60 else "")
            # Отсекаем только продолжение серии; первая реплика всегда сохраняется.
            if is_live_spam(text, repeat_streak):
                live_dropped += 1
                if bar:
                    bar.update(max(0, int(seg.end) - bar.n))
                elif progress:
                    progress(seg.end, snippet)
                continue

            avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            low_conf = avg_logprob < p.low_confidence_logprob or no_speech_prob > 0.6
            if low_conf:
                uncertain += 1
                review_lines.append(
                    f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] "
                    f"logprob={avg_logprob:.2f}, no_speech={no_speech_prob:.2f}\n{text}\n"
                )

            accepted_segment = Segment(
                float(seg.start), float(seg.end), text, avg_logprob, no_speech_prob
            )
            checkpoint.append(accepted_segment)

            if bar:
                bar.set_postfix_str(snippet)
                bar.update(max(0, int(seg.end) - bar.n))
            elif progress:
                progress(float(seg.end), snippet)
    except InterruptedError:
        _log(log, f"Прогресс сохранён: {paths['checkpoint'].name}")
        raise
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        _log(log, f"Причина сбоя: {reason}")
        _log(log, f"Прогресс сохранён: {paths['checkpoint'].name}")
        raise RuntimeError(
            f"Транскрибация прервана на {format_ts(checkpoint.resume_at)}. "
            f"Повторный запуск продолжит работу. Причина: {reason}"
        ) from exc
    finally:
        checkpoint.close()
        if bar:
            bar.close()

    if progress and duration:
        progress(duration, "Готово")

    if live_dropped:
        _log(log, f"Отфильтровано на лету (повторы): {live_dropped} сегментов")

    if config.save_raw:
        raw_header = (
            f"# Raw transcript | {audio.name} | preset={config.preset} | model={config.model}\n"
        )
        _write_text(paths["raw_txt"], segments_to_txt(raw_segments, raw_header))
        _log(log, f"Сохранено: {paths['raw_txt'].name} ({len(raw_segments)} сегментов)")

    if config.postprocess:
        final_segments, clean_log = clean_segments(raw_segments, audio.name)
        _write_text(paths["log_txt"], "\n".join(clean_log))
        _log(
            log,
            f"Постобработка: {len(raw_segments)} -> {len(final_segments)} сегментов",
        )
    else:
        final_segments = raw_segments
        _write_text(paths["log_txt"], "# Постобработка отключена пользователем\n")

    header = f"# Транскрипт | {audio.name} | preset={config.preset}\n"
    _write_text(paths["final_txt"], segments_to_txt(final_segments, header))
    _log(log, f"Сохранено: {paths['final_txt'].name}")

    if config.save_srt:
        _write_text(paths["final_srt"], segments_to_srt(final_segments))
        _log(log, f"Сохранено: {paths['final_srt'].name}")

    if config.save_plain:
        _write_text(paths["final_plain"], segments_to_plain(final_segments))
        _log(log, f"Сохранено: {paths['final_plain'].name}")

    if config.save_review:
        review_content = (
            "# Сегменты для ручной проверки\n\n" + "\n".join(review_lines)
            if review_lines
            else "# Сегменты для ручной проверки\n\nСомнительных сегментов не найдено.\n"
        )
        _write_text(paths["review_txt"], review_content)
        _log(log, f"Сохранено: {paths['review_txt'].name} ({uncertain} сомнительных)")

    checkpoint.complete()
    _log(log, "Контрольная точка завершена и удалена.")
    _log(log, "\nГотово.")
    return paths
