"""Пресеты и конфиг одного прогона транскрибации."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, replace
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

# Контекст, специфичный для одной записи, искажает распознавание остальных файлов.
# Пользователь может заполнить эти поля явно в GUI или CLI.
DEFAULT_PROMPT = ""
DEFAULT_HOTWORDS = ""

_INVALID_STEM_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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
    device: str = "auto"
    device_index: int = 0
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
    resume: bool = True
    clip_start: float | None = None
    clip_end: float | None = None

    def apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            raise ValueError(f"Неизвестный пресет: {name}. Доступны: {', '.join(PRESETS)}")
        p = PRESETS[name]
        self.preset = name
        self.model = p["model"]
        self.compute_type = p["compute_type"]
        # У каждого запуска должен быть независимый набор параметров.
        self.params = replace(p["params"])

    @classmethod
    def from_preset(cls, audio_path: Path, preset: str = "safe", **kwargs) -> TranscribeConfig:
        cfg = cls(audio_path=Path(audio_path))
        cfg.apply_preset(preset)
        valid_fields = {f.name for f in fields(cls)}
        for key, val in kwargs.items():
            if key not in valid_fields:
                raise TypeError(f"Неизвестный параметр конфигурации: {key}")
            setattr(cfg, key, val)
        return cfg

    @classmethod
    def from_dict(cls, data: dict) -> TranscribeConfig:
        """Безопасно восстанавливает конфигурацию из JSON-совместимого словаря."""
        if not isinstance(data, dict):
            raise ValueError("Конфигурация должна быть JSON-объектом")
        if not data.get("audio_path"):
            raise ValueError("В конфигурации отсутствует audio_path")

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Неизвестные поля конфигурации: {', '.join(sorted(unknown))}")

        values = dict(data)
        values["audio_path"] = Path(values["audio_path"])
        if values.get("output_dir"):
            values["output_dir"] = Path(values["output_dir"])
        params_data = values.pop("params", None)
        cfg = cls(**values)
        if params_data is not None:
            if not isinstance(params_data, dict):
                raise ValueError("params должен быть JSON-объектом")
            param_names = {f.name for f in fields(TranscribeParams)}
            unknown_params = set(params_data) - param_names
            if unknown_params:
                raise ValueError(
                    f"Неизвестные параметры транскрибации: {', '.join(sorted(unknown_params))}"
                )
            cfg.params = TranscribeParams(**params_data)
        cfg.validate()
        return cfg

    def validate(self, *, check_audio: bool = False) -> None:
        """Проверяет конфигурацию до загрузки тяжёлой модели."""
        self.audio_path = Path(self.audio_path)
        if check_audio and not self.audio_path.expanduser().is_file():
            resolved_audio = self.audio_path.expanduser().resolve()
            raise FileNotFoundError(f"Аудиофайл не найден: {resolved_audio}")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.preset not in PRESETS:
            raise ValueError(f"Неизвестный пресет: {self.preset}")
        if not self.model or not self.model.strip():
            raise ValueError("Название модели не может быть пустым")
        if self.device not in DEVICES:
            raise ValueError(f"Устройство должно быть одним из: {', '.join(DEVICES)}")
        if self.device_index < 0:
            raise ValueError("device_index не может быть отрицательным")
        if self.compute_type not in COMPUTE_TYPES:
            raise ValueError(f"compute_type должен быть одним из: {', '.join(COMPUTE_TYPES)}")
        if not self.language or not self.language.strip():
            raise ValueError("Код языка не может быть пустым")

        if self.stem is not None:
            stem = self.stem.strip()
            if not stem or stem in {".", ".."} or _INVALID_STEM_RE.search(stem):
                raise ValueError("Префикс имени файла содержит недопустимые символы")
            if stem.endswith((" ", ".")):
                raise ValueError("Префикс имени файла не должен оканчиваться пробелом или точкой")
            self.stem = stem

        p = self.params
        if p.beam_size < 1 or p.best_of < 1:
            raise ValueError("beam_size и best_of должны быть не меньше 1")
        if p.patience <= 0:
            raise ValueError("patience должен быть больше 0")
        if p.temperature < 0:
            raise ValueError("temperature не может быть отрицательной")
        for name in ("vad_threshold", "no_speech_threshold"):
            value = getattr(p, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} должен быть в диапазоне от 0 до 1")
        if p.vad_min_speech_ms < 0 or p.vad_min_silence_ms < 0:
            raise ValueError("Длительности VAD не могут быть отрицательными")
        if self.clip_start is not None and self.clip_start < 0:
            raise ValueError("clip_start не может быть отрицательным")
        if self.clip_end is not None:
            if self.clip_start is None:
                raise ValueError("clip_end требует указать clip_start")
            if self.clip_end <= self.clip_start:
                raise ValueError("clip_end должен быть больше clip_start")

    def resolved_output_dir(self) -> Path:
        return (self.output_dir or self.audio_path.parent).expanduser()

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
            "checkpoint": out / f"{stem}_transcript.partial.jsonl",
        }

    def effective_compute_type(self) -> str:
        if self.compute_type == "default":
            return PRESETS.get(self.preset, PRESETS["safe"])["compute_type"]
        return self.compute_type

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "params"}
        d["params"] = asdict(self.params)
        d["audio_path"] = str(self.audio_path)
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        return d
