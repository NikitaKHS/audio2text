"""CLI для высокоточной русской транскрибации GigaAM-v3 с двойной проверкой."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from a2t_lib.accuracy import transcribe_high_accuracy
from a2t_lib.timestamps import format_ts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Высокоточная локальная RU-транскрибация двумя декодерами GigaAM-v3"
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Один или несколько аудиофайлов")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--corrections", type=Path, help="JSON с подтверждёнными исправлениями")
    parser.add_argument("--agreement-threshold", type=float, default=0.86)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.agreement_threshold <= 1:
        print("agreement-threshold должен быть от 0 до 1", file=sys.stderr)
        return 2
    for audio in args.audio:
        print(f"\n=== {audio.name}: высокоточная RU-транскрибация ===", flush=True)

        def report(done: int, total: int, seconds: float) -> None:
            if done == total or done == 1 or done % 10 == 0:
                print(
                    f"{done}/{total} ({done / total:.0%}) · до {format_ts(seconds)}",
                    flush=True,
                )

        try:
            paths = transcribe_high_accuracy(
                audio,
                output_dir=args.output_dir,
                corrections_path=args.corrections,
                agreement_threshold=args.agreement_threshold,
                progress=report,
            )
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        print(f"Готово: {paths['txt']}")
        print(f"Проверить: {paths['review']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
