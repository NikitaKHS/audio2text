"""CLI. Обёртка над a2t_lib.engine.transcribe_file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from a2t_lib.config import COMPUTE_TYPES, DEVICES, MODELS, PRESETS, TranscribeConfig


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
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--device", choices=DEVICES, default="auto")
    p.add_argument("--device-index", type=int, default=0, help="Индекс GPU для нескольких карт")
    p.add_argument(
        "--compute-type",
        choices=COMPUTE_TYPES,
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
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Продолжить совместимую незавершённую транскрибацию",
    )
    p.add_argument("--list-presets", action="store_true", help="Показать пресеты и выйти")
    p.add_argument("--system-info", action="store_true", help="Показать найденные GPU и выйти")
    p.add_argument("--save-config", type=Path, help="Сохранить итоговый конфиг в JSON")
    p.add_argument("--load-config", type=Path, help="Загрузить конфиг из JSON")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Проверить конфигурацию без загрузки модели и транскрибации",
    )
    p.add_argument("--debug", action="store_true", help="Показать traceback при ошибке")
    return p


def config_from_args(args: argparse.Namespace) -> TranscribeConfig:
    if args.load_config:
        try:
            data = json.loads(args.load_config.read_text(encoding="utf-8-sig"))
        except FileNotFoundError as exc:
            raise SystemExit(f"Файл конфигурации не найден: {args.load_config}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Не удалось прочитать конфигурацию: {exc}") from exc
        try:
            return TranscribeConfig.from_dict(data)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Некорректная конфигурация: {exc}") from exc

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
    cfg.device_index = args.device_index
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
    cfg.resume = args.resume
    cfg.clip_start = args.clip_start
    cfg.clip_end = args.clip_end
    try:
        cfg.validate()
    except ValueError as exc:
        raise SystemExit(f"Некорректные параметры: {exc}") from exc
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_presets:
        for name, meta in PRESETS.items():
            print(f"{name}: {meta['description']}")
            print(f"  model={meta['model']}, compute={meta['compute_type']}")
        return

    if args.system_info:
        from a2t_lib.hardware import detect_gpus

        gpus = detect_gpus()
        if not gpus:
            print("NVIDIA GPU не найден. Будет использован CPU (int8).")
        for gpu in gpus:
            print(gpu.summary())
            if gpu.driver:
                print(f"  driver={gpu.driver}, compute_capability={gpu.compute_capability}")
            if gpu.compute_types:
                print(f"  compute_types={', '.join(gpu.compute_types)}")
        return

    cfg = config_from_args(args)

    if args.save_config:
        args.save_config.parent.mkdir(parents=True, exist_ok=True)
        args.save_config.write_text(
            json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Конфиг сохранён: {args.save_config}")

    if args.validate_only:
        try:
            cfg.validate(check_audio=True)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            sys.exit(1)
        print("Конфигурация корректна.")
        return

    try:
        # Тяжёлый ML runtime не нужен для --help, --list-presets и проверки JSON.
        from a2t_lib.engine import transcribe_file

        transcribe_file(cfg)
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if args.debug:
            raise
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
