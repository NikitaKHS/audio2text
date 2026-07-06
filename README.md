# audio2text

Транскрибация аудио на русском через [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3 на GPU). Есть GUI, CLI и постобработка от типичного мусора Whisper — зацикленные однословные сегменты, «спасибо за просмотр», повторы в тексте.

Собрано под длинные записи (интервью, терапия, подкасты). Для русского длинного аудио берите пресет **safe**, не **fast**.

## Требования

- Windows 10/11 (проверялось здесь; на Linux/macOS должно завестись без bat-файлов)
- Python 3.11+ (`py -3`)
- NVIDIA GPU с CUDA — для нормальной скорости; CPU работает, но медленно
- [ffmpeg](https://ffmpeg.org/) в PATH или `C:\ffmpeg\bin\ffmpeg.exe`

## Установка

```powershell
cd audio2text
py -3 -m pip install -r requirements.txt
```

Первый запуск скачает модель Whisper в `%USERPROFILE%\.cache\huggingface\`.

## Запуск

**GUI** (основной способ):

```powershell
run_gui.bat
# или
py -3 transcribe_gui.py
```

**CLI**:

```powershell
py -3 transcribe_cli.py путь\к\записи.m4a
py -3 transcribe_cli.py запись.m4a -o results --preset safe
py -3 transcribe_cli.py --list-presets
py -3 transcribe_cli.py --help
```

**Старый скрипт** `transcribe.py` — обрабатывает все аудио в папке, пресет safe. Оставлен для совместимости.

## Пресеты

| Пресет | Модель | Когда |
|--------|--------|--------|
| `safe` | large-v3, float16 | Длинные записи, русский — **дефолт** |
| `balanced` | large-v3, float16 | Чуть быстрее, чуть рискованнее |
| `max_quality` | large-v3, float32 | Максимум качества, ~10+ ГБ VRAM, долго |
| `fast` | distil-large-v3, int8 | Черновик; на русском часто хуже |

## Выходные файлы

Для `session.m4a` в папке вывода:

| Файл | Содержимое |
|------|------------|
| `session_final.txt` | Транскрипт с таймкодами |
| `session_final.srt` | Субтитры |
| `session_final_plain.txt` | Текст без меток |
| `session_transcript_raw.txt` | До постобработки |
| `session_corrections_log.txt` | Что выкинули/поправили |
| `session_review.txt` | Сегменты с низкой уверенностью (если есть) |

## Структура проекта

```
audio2text/
  a2t_lib/             # ядро
    config.py
    engine.py
    postprocess.py
    gui_widgets.py
    timestamps.py
  transcribe_gui.py
  transcribe_cli.py
  transcribe.py        # legacy
  tests/
  run_gui.bat
  run_tests.bat
```

Скрипты `transcribe_25555.py`, `clean_25555.py`, `fix_25555_*.py`, `manual_pass*.py` — разовые правки конкретной записи, в общий пайплайн не входят.

## Тесты

```powershell
run_tests.bat
# или
py -3 -m unittest discover -s tests -v
```

36 тестов: конфиг, постобработка, CLI, движок с мок-моделью, smoke GUI.

Короткий прогон на GPU (15 сек из m4a):

```powershell
ffmpeg -y -i session.m4a -t 15 _smoke_test\clip.m4a
py -3 transcribe_cli.py _smoke_test\clip.m4a -o _smoke_test --stem test --preset safe
```

## Заметки

- **`condition_on_previous_text=False`** по умолчанию — иначе на длинных паузах Whisper иногда зацикливается.
- **`--clip-start` / `--clip-end`** — faster-whisper режет аудио, но в логе может показывать полную длительность файла; для точного теста лучше нарезать ffmpeg.
- Консоль Windows (cp1251) не любит некоторые символы в print — в коде используем ASCII-стрелки в логах.
- GUI крутит транскрибацию в фоновом потоке; отмена — best-effort, модель может договорить текущий кусок.

## Лицензия

Код — как есть, для личного использования. Whisper / faster-whisper — свои лицензии.
