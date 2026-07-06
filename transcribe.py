"""Legacy: все аудио в папке скрипта, пресет safe. Лучше transcribe_cli.py / GUI."""
from pathlib import Path
import sys

from a2t_lib.config import TranscribeConfig
from a2t_lib.engine import transcribe_file

FOLDER = Path(__file__).resolve().parent
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".webm", ".mp4", ".mkv"}


def find_audio(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        sys.exit("No audio files found.")
    if len(files) > 1:
        print("Несколько файлов — обрабатываю все:\n  " + "\n  ".join(f.name for f in files))
    return files


def main() -> None:
    for audio in find_audio(FOLDER):
        cfg = TranscribeConfig.from_preset(audio, preset="safe")
        transcribe_file(cfg)
    print("\nГотово.")


if __name__ == "__main__":
    main()
