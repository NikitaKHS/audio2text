"""Фильтрация сегментов после Whisper: spam, loops, junk."""

from __future__ import annotations

import re
from dataclasses import dataclass

from a2t_lib.timestamps import format_ts, ts_srt

TS_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2}\.\d+) -> (\d{2}):(\d{2}):(\d{2}\.\d+)\]\s*(.*)",
    re.DOTALL,
)

# одно слово, которое Whisper часто печатает сотнями раз на тишине
SINGLE_WORD_SPAM = {
    "умереть",
    "умри",
    "умрите",
    "спасибо",
    "продолжение следует",
}

STRUCTURAL_JUNK_REASONS = {
    "зацикленные hotwords",
    "повторяющийся фрагмент",
    "зацикленное слово",
}

JUNK_PHRASES = re.compile(
    r"(спасибо за просмотр|subscrib|субтитр|amara\.org|doram|"
    r"продолжение следует|не забудьте подписаться|"
    r"thanks for watching|please subscribe)",
    re.I,
)

HOTWORD_SPAM = re.compile(
    r"(?:Выборг|Лебедевка|Гаврилова|аэрофобия|терапия|Рахат[- ]?лук[ао]м|"
    r"Данил|Петербург)(?:,\s*(?:Выборг|Лебедевка|Гаврилова|аэрофобия|"
    r"терапия|Рахат[- ]?лук[ао]м|Данил|Петербург)){4,}",
    re.I,
)

LOOP_PATTERNS = [
    (r"(,\s*и потом,\s*вот)(?:\s*,\s*и потом,\s*вот)+", ""),
    (r"(,\s*ну,\s*я)(?:,\s*ну,\s*я)+", ""),
    (r"(,\s*она)(?:,\s*она)+", ", она"),
    (r"(,\s*я не знаю)(?:,\s*я не знаю)+", ", я не знаю"),
    (r"(,\s*я не знаю),\s*я(?=\s*$|[,.])", r"\1"),
    (r"(,\s*и так)(?:,\s*и так)+", ""),
    (r"(,\s*и все)(?:,\s*и все)+", ""),
    (r"(,\s*это)(?:,\s*это)+", ", это"),
    (r"и так далее, и так далее, и так далее, и", "и так далее, и"),
]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


def parse_ts(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def norm(text: str) -> str:
    return text.strip().lower().rstrip(".!?…")


def remove_loops(text: str) -> str:
    for pat, repl in LOOP_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return re.sub(r",\s*,", ", ", text)


def is_single_word_spam(text: str) -> bool:
    return norm(text) in SINGLE_WORD_SPAM


def keep_legitimate_context(segments: list[Segment], idx: int) -> bool:
    """Короткий ответ после вопроса о страхе — оставляем (не путать с spam-петлёй)."""
    if not is_single_word_spam(segments[idx].text):
        return False
    prev = segments[idx - 1] if idx > 0 else None
    if not prev or segments[idx].start - prev.end > 12:
        return False
    prev_text = prev.text.lower()
    if not (
        "?" in prev.text
        or any(w in prev_text for w in ("боял", "страш", "чего боя", "опас", "риск"))
    ):
        return False
    if idx > 0 and is_single_word_spam(segments[idx - 1].text):
        return False
    return True


def is_junk(seg: Segment) -> tuple[bool, str]:
    text = seg.text.strip()
    if not text:
        return True, "пустой сегмент"
    if JUNK_PHRASES.search(text):
        return True, "типичная галлюцинация Whisper"
    if HOTWORD_SPAM.search(text):
        return True, "зацикленные hotwords"
    if len(text) < 80 and text.count(",") > 8 and len(set(text.split(","))) < 4:
        return True, "повторяющийся фрагмент"
    words = text.split()
    if len(words) >= 6 and len(set(w.lower() for w in words)) <= 2:
        return True, "зацикленное слово"
    return False, ""


def is_live_spam(text: str, repeat_streak: int, threshold: int = 3) -> bool:
    """Тот же принцип, что в clean_segments, но во время streaming."""
    n = norm(text)
    if n in SINGLE_WORD_SPAM and repeat_streak >= threshold:
        return True
    if repeat_streak >= 4 and len(text.split()) <= 2:
        return True
    return False


def clean_segments(
    segments: list[Segment], source_name: str = "raw"
) -> tuple[list[Segment], list[str]]:
    log: list[str] = [f"# Постобработка — {source_name}\n"]
    removed = 0
    spam_removed = 0
    kept: list[Segment] = []
    previous_norm = ""
    repeat_streak = 0

    for i, seg in enumerate(segments):
        current_norm = norm(seg.text)
        if current_norm and current_norm == previous_norm:
            repeat_streak += 1
        else:
            repeat_streak = 1 if current_norm else 0
            previous_norm = current_norm

        if is_single_word_spam(seg.text):
            # Одиночные «спасибо» и «умереть» могут быть реальными репликами.
            # Удаляем только доказанный последовательный повтор, сохраняя первый.
            if repeat_streak >= 3 and not keep_legitimate_context(segments, i):
                removed += 1
                spam_removed += 1
                if spam_removed <= 5:
                    log.append(
                        f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] "
                        f"Удалено: повторяющийся однословный сегмент\n"
                    )
                continue

        junk, reason = is_junk(seg)
        low_confidence = seg.avg_logprob < -0.55 or seg.no_speech_prob > 0.6
        if junk and (reason in STRUCTURAL_JUNK_REASONS or low_confidence):
            removed += 1
            log.append(
                f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] "
                f"Удалено ({reason}): {seg.text[:100]}{'…' if len(seg.text) > 100 else ''}\n"
            )
            continue
        if junk:
            log.append(
                f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] "
                f"Сохранено для проверки ({reason}, уверенность нормальная): {seg.text[:100]}"
                f"{'…' if len(seg.text) > 100 else ''}\n"
            )

        cleaned = remove_loops(seg.text)
        if cleaned != seg.text:
            log.append(
                f"[{format_ts(seg.start)} -> {format_ts(seg.end)}] Убран loop: …{seg.text[-80:]}\n"
            )
        if cleaned:
            kept.append(Segment(seg.start, seg.end, cleaned, seg.avg_logprob, seg.no_speech_prob))

    if spam_removed > 5:
        log.insert(5, f"... ещё {spam_removed - 5} однословных повторов удалено\n")

    log.append(
        f"\n## Итого\nБыло сегментов: {len(segments)}\nСтало: {len(kept)}\nУдалено: {removed}\n"
    )
    if kept:
        log.append(f"Покрытие: {format_ts(kept[0].start)} — {format_ts(kept[-1].end)}\n")
    return kept, log


def parse_transcript_file(path) -> list[Segment]:
    """Читает наш формат [HH:MM:SS -> HH:MM:SS] text."""
    from pathlib import Path

    segments: list[Segment] = []
    for block in Path(path).read_text(encoding="utf-8").strip().split("\n\n"):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        m = TS_RE.match(block)
        if not m:
            continue
        segments.append(
            Segment(
                parse_ts(m.group(1), m.group(2), m.group(3)),
                parse_ts(m.group(4), m.group(5), m.group(6)),
                m.group(7).strip(),
            )
        )
    return segments


def segments_to_txt(segments: list[Segment], header: str = "") -> str:
    body = "\n\n".join(f"[{format_ts(s.start)} -> {format_ts(s.end)}] {s.text}" for s in segments)
    return f"{header}\n\n{body}" if header else body


def segments_to_srt(segments: list[Segment]) -> str:
    return "\n".join(
        f"{i}\n{ts_srt(s.start)} --> {ts_srt(s.end)}\n{s.text}\n" for i, s in enumerate(segments, 1)
    )


def segments_to_plain(segments: list[Segment]) -> str:
    return "\n".join(s.text for s in segments)
