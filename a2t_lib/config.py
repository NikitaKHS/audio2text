"""Пресеты и конфиг одного прогона транскрибации."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]

DEVICES = ["cuda", "cpu", "auto"]
COMPUTE_TYPES = ["default", "float32", "float16", "int8", "int8_float16", "bfloat16"]

DEFAULT_PROMPT = (
    "Психотерапевтическая сессия. Разговор на русском языке. "
    "Терапия, аэрофобия, одиночество, отношения."
)
DEFAULT_HOTWORDS = "терапия аэрофобия психотерапия"


@dataclass
class TranscribeParams:
    """Параметры, которые уходят в model.transcribe()."""

    beam_size: int = 5
    best_of: int = 1
    patience: float = 1.0
    temperature: float = 0.0
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 600
    word_timestamps: bool = False
    # False — меньше зацикливаний на тишине; True — иногда лучше связность
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.75
    compression_ratio_threshold: float = 2.2
    hallucination_silence_threshold: float = 1.5
    low_confidence_logprob: float = -0.55


PRESETS: dict[str, dict] = {
    "safe": {
        "description": "Надёжный режим — защита от повторов и галлюцинаций",
        "model": "large-v3",
        "compute_type": "float16",
        "params": TranscribeParams(
            beam_size=5,
            best_of=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            hallucination_silence_threshold=1.5,
            compression_ratio_threshold=2.2,
        ),
    },
    "balanced": {
        "description": "Баланс скорости и качества",
        "model": "large-v3",
        "compute_type": "float16",
        "params": TranscribeParams(
            beam_size=5,
            best_of=3,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            hallucination_silence_threshold=1.8,
        ),
    },
    "max_quality": {
        "description": "Максимум качества (медленно, ~10+ ГБ VRAM)",
        "model": "large-v3",
        "compute_type": "float32",
        "params": TranscribeParams(
            beam_size=10,
            best_of=5,
            patience=2.0,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.65,
            hallucination_silence_threshold=2.0,
            compression_ratio_threshold=2.4,
        ),
    },
    "fast": {
        "description": "Быстро — distil-large-v3, int8",
        "model": "distil-large-v3",
        "compute_type": "int8_float16",
        "params": TranscribeParams(
            beam_size=3,
            best_of=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.78,
        ),
    },
}


@dataclass
class TranscribeConfig:
    audio_path: Path
    output_dir: Path | None = None
    stem: str | None = None
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ru"
    preset: str = "safe"
    initial_prompt: str = DEFAULT_PROMPT
    hotwords: str = DEFAULT_HOTWORDS
    params: TranscribeParams = field(default_factory=TranscribeParams)
    postprocess: bool = True
    save_srt: bool = True
    save_plain: bool = True
    save_raw: bool = True
    save_review: bool = True
    clip_start: float | None = None
    clip_end: float | None = None

    def apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            raise ValueError(f"Неизвестный пресет: {name}. Доступны: {', '.join(PRESETS)}")
        p = PRESETS[name]
        self.preset = name
        self.model = p["model"]
        self.compute_type = p["compute_type"]
        self.params = p["params"]

    @classmethod
    def from_preset(cls, audio_path: Path, preset: str = "safe", **kwargs) -> TranscribeConfig:
        cfg = cls(audio_path=Path(audio_path))
        cfg.apply_preset(preset)
        for key, val in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.audio_path.parent

    def resolved_stem(self) -> str:
        return self.stem or self.audio_path.stem

    def output_paths(self) -> dict[str, Path]:
        out = self.resolved_output_dir()
        stem = self.resolved_stem()
        return {
            "final_txt": out / f"{stem}_final.txt",
            "final_srt": out / f"{stem}_final.srt",
            "final_plain": out / f"{stem}_final_plain.txt",
            "raw_txt": out / f"{stem}_transcript_raw.txt",
            "review_txt": out / f"{stem}_review.txt",
            "log_txt": out / f"{stem}_corrections_log.txt",
        }

    def effective_compute_type(self) -> str:
        if self.compute_type == "default":
            return PRESETS.get(self.preset, PRESETS["safe"])["compute_type"]
        return self.compute_type

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "params"}
        d["params"] = self.params.__dict__
        d["audio_path"] = str(self.audio_path)
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        return d
