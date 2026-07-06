"""CLI. Обёртка над a2t_lib.engine.transcribe_file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from a2t_lib.config import PRESETS, TranscribeConfig, TranscribeParams
from a2t_lib.engine import transcribe_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Транскрибация аудио через faster-whisper (audio2text)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Путь к аудиофайлу (.m4a, .mp3, .wav, …)",
    )
    p.add_argument("-o", "--output-dir", type=Path, help="Папка для результатов")
    p.add_argument("--stem", help="Префикс имён выходных файлов (по умолчанию — имя аудио)")
    p.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="safe",
        help="Пресет качества/скорости",
    )
    p.add_argument("--model", choices=[
        "tiny", "base", "small", "medium", "large-v2",
        "large-v3", "large-v3-turbo", "distil-large-v3",
    ])
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument(
        "--compute-type",
        choices=["default", "float32", "float16", "int8", "int8_float16", "bfloat16"],
        default="default",
    )
    p.add_argument("--language", default="ru")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--best-of", type=int)
    p.add_argument("--no-speech-threshold", type=float)
    p.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--hotwords", default=None)
    p.add_argument("--clip-start", type=float, help="Начало фрагмента (сек)")
    p.add_argument("--clip-end", type=float, help="Конец фрагмента (сек)")
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--no-srt", action="store_true")
    p.add_argument("--no-plain", action="store_true")
    p.add_argument("--no-raw", action="store_true")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--list-presets", action="store_true", help="Показать пресеты и выйти")
    p.add_argument("--save-config", type=Path, help="Сохранить итоговый конфиг в JSON")
    p.add_argument("--load-config", type=Path, help="Загрузить конфиг из JSON")
    return p


def config_from_args(args: argparse.Namespace) -> TranscribeConfig:
    if args.load_config:
        data = json.loads(args.load_config.read_text(encoding="utf-8"))
        cfg = TranscribeConfig(
            audio_path=Path(data["audio_path"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            stem=data.get("stem"),
            model=data.get("model", "large-v3"),
            device=data.get("device", "cuda"),
            compute_type=data.get("compute_type", "float16"),
            language=data.get("language", "ru"),
            preset=data.get("preset", "safe"),
            initial_prompt=data.get("initial_prompt", ""),
            hotwords=data.get("hotwords", ""),
            postprocess=data.get("postprocess", True),
            save_srt=data.get("save_srt", True),
            save_plain=data.get("save_plain", True),
            save_raw=data.get("save_raw", True),
            save_review=data.get("save_review", True),
            clip_start=data.get("clip_start"),
            clip_end=data.get("clip_end"),
        )
        if "params" in data:
            cfg.params = TranscribeParams(**data["params"])
        return cfg

    if not args.audio:
        raise SystemExit("Укажите путь к аудиофайлу или --load-config")

    cfg = TranscribeConfig.from_preset(args.audio, preset=args.preset)

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.stem:
        cfg.stem = args.stem
    if args.model:
        cfg.model = args.model
    cfg.device = args.device
    if args.compute_type != "default":
        cfg.compute_type = args.compute_type
    cfg.language = args.language
    if args.initial_prompt is not None:
        cfg.initial_prompt = args.initial_prompt
    if args.hotwords is not None:
        cfg.hotwords = args.hotwords
    if args.beam_size is not None:
        cfg.params.beam_size = args.beam_size
    if args.best_of is not None:
        cfg.params.best_of = args.best_of
    if args.no_speech_threshold is not None:
        cfg.params.no_speech_threshold = args.no_speech_threshold
    if args.condition_on_previous_text is not None:
        cfg.params.condition_on_previous_text = args.condition_on_previous_text
    cfg.postprocess = not args.no_postprocess
    cfg.save_srt = not args.no_srt
    cfg.save_plain = not args.no_plain
    cfg.save_raw = not args.no_raw
    cfg.save_review = not args.no_review
    cfg.clip_start = args.clip_start
    cfg.clip_end = args.clip_end
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_presets:
        for name, meta in PRESETS.items():
            print(f"{name}: {meta['description']}")
            print(f"  model={meta['model']}, compute={meta['compute_type']}")
        return

    cfg = config_from_args(args)

    if args.save_config:
        args.save_config.write_text(
            json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Конфиг сохранён: {args.save_config}")

    try:
        transcribe_file(cfg)
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
