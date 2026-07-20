# audio2text studio

Локальное приложение для точной транскрибации длинных аудио- и видеозаписей. Поддерживает
быстрый Whisper large-v3 на NVIDIA GPU и отдельный высокоточный русский режим
GigaAM-v3 RNNT + CTC с двойной проверкой каждого речевого фрагмента.

[![CI](https://github.com/NikitaKHS/audio2text/actions/workflows/ci.yml/badge.svg)](https://github.com/NikitaKHS/audio2text/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

## Возможности

- Два понятных режима в одном GUI: `Whisper GPU` и `GigaAM-v3 RNNT + CTC`.
- Выбор конкретной NVIDIA GPU, проверка CUDA 12, cuBLAS и cuDNN 9 до запуска.
- Надёжная обработка многочасовых M4A, MP3, WAV, OGG, FLAC, WEBM, MP4 и MKV.
- Автоматические контрольные точки: прерванная задача продолжается с готового фрагмента.
- TXT с таймкодами, обычный TXT, SRT, raw-версия, журнал очистки и review-файл.
- Защита от повторов, тишины и типичных галлюцинаций распознавания.
- Аудио обрабатывается локально и не отправляется во внешние сервисы.

## Какой режим выбрать

| Режим | Когда использовать | Устройство | Результат |
|---|---|---|---|
| **Whisper GPU** | Быстрый черновик, разные языки, большие объёмы | NVIDIA GPU или CPU | `*_final.*` |
| **GigaAM-v3 RNNT + CTC** | Важная русская разговорная речь | CPU; GPU при CUDA-сборке PyTorch | `*_high_accuracy.*` |

В режиме GigaAM RNNT создаёт основной текст, а CTC повторно распознаёт тот же фрагмент. Если
гипотезы заметно расходятся, участок попадает в `*_high_accuracy_review.txt`.

```text
Аудио → VAD → GigaAM RNNT → GigaAM CTC → сверка → TXT + SRT + REVIEW
```

## Быстрый старт на Windows

Требуется Windows 10/11 и Python 3.10 или новее.

```powershell
git clone https://github.com/NikitaKHS/audio2text.git
cd audio2text
setup.bat
run_gui.bat
```

Дальше в приложении:

1. Выберите аудиофайл.
2. Нажмите карточку `Whisper GPU` или `GigaAM-v3 RNNT + CTC`.
3. Проверьте папку сохранения.
4. Запустите транскрибацию.
5. Нажмите `Открыть текст` или `Открыть результаты`.

Горячие клавиши: `Ctrl+O` — выбрать файл, `Ctrl+Enter` — запуск, `Esc` — безопасная отмена.

## Установка GigaAM-v3

Нажмите `Установить GigaAM-v3` прямо в приложении либо выполните:

```powershell
setup_accuracy.bat
```

Установщик добавляет PyTorch и закреплённую версию официального
[GigaAM](https://github.com/salute-developers/GigaAM). Модели RNNT и CTC загружаются только при
первом запуске, после чего используются из локального кэша.

## Результаты

### Whisper

| Файл | Содержимое |
|---|---|
| `recording_final.txt` | Итоговый текст с таймкодами |
| `recording_final_plain.txt` | Текст без таймкодов |
| `recording_final.srt` | Субтитры |
| `recording_transcript_raw.txt` | Результат до очистки |
| `recording_review.txt` | Фрагменты с низкой уверенностью |
| `recording_corrections_log.txt` | Журнал автоматической очистки |

### GigaAM-v3 RNNT + CTC

| Файл | Содержимое |
|---|---|
| `recording_high_accuracy.txt` | Проверенный двумя декодерами текст с таймкодами |
| `recording_high_accuracy_plain.txt` | Текст без таймкодов |
| `recording_high_accuracy.srt` | Субтитры |
| `recording_high_accuracy_review.txt` | Места расхождения RNNT и CTC |
| `recording_high_accuracy_corrections.txt` | Применённые ручные замены и отброшенный мусор |

Файлы `*.partial.jsonl` — временные контрольные точки. После успешного завершения они удаляются.

## Подтверждённые ручные исправления

Если конкретная фраза проверена по аудио, её можно исправить без глобальной автозамены. Скопируйте
`accuracy_corrections.example.json`, заполните свой JSON и выберите его в интерфейсе GigaAM.

```json
{
  "recording.m4a": [
    {
      "start": 5.0,
      "end": 17.0,
      "find": "ошибочная фраза",
      "replace": "верная фраза",
      "note": "проверено по аудио"
    }
  ]
}
```

Замена применяется только в указанном временном диапазоне и записывается в журнал.

## CLI

Whisper:

```powershell
.\.venv\Scripts\python.exe transcribe_cli.py recording.m4a --preset safe
.\.venv\Scripts\python.exe transcribe_cli.py --system-info
.\.venv\Scripts\python.exe transcribe_cli.py --list-presets
```

GigaAM-v3 для одного или нескольких файлов:

```powershell
.\.venv\Scripts\python.exe transcribe_accuracy.py 1.m4a 2.m4a
.\.venv\Scripts\python.exe transcribe_accuracy.py 1.m4a --corrections corrections.json
```

Все параметры CLI доступны через `--help`.

## Whisper-пресеты

| Пресет | Модель | Назначение |
|---|---|---|
| `safe` | large-v3, float16 | Длинные и важные записи; рекомендуется |
| `balanced` | large-v3, float16 | Баланс скорости и качества |
| `max_quality` | large-v3, float32 | Требуется обычно более 10 ГБ VRAM |
| `fast` | distil-large-v3, int8 | Быстрый черновик |

Для RTX 3080 Laptop с 8 ГБ VRAM рекомендуется:

```text
safe · large-v3 · cuda:0 · float16
```

## Надёжность и точность

- Выходные файлы заменяются атомарно и не остаются обрезанными после сбоя.
- Checkpoint проверяет путь, размер, время изменения файла и параметры моделей.
- VAD отделяет речь от длинной тишины.
- GigaAM сравнивает две гипотезы и показывает расхождения вместо ложной гарантии качества.
- Точный WER можно подтвердить только сравнением с ручным эталонным текстом.

## Разработка

```powershell
run_tests.bat
```

Локальная проверка:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
```

CI проверяет Python 3.10 и 3.13 на Windows.

Основные модули:

```text
a2t_lib/
  accuracy.py     GigaAM RNNT + CTC, VAD, checkpoint и сверка
  checkpoint.py   восстановление длинных Whisper-задач
  config.py       конфигурация и пресеты
  engine.py       faster-whisper и сохранение результатов
  hardware.py     GPU и CUDA runtime
  postprocess.py  очистка повторов и галлюцинаций
  runtime.py      защита Windows от сна
transcribe_gui.py
transcribe_cli.py
transcribe_accuracy.py
```

## Автор

[NikitaKHS](https://github.com/NikitaKHS)

## Лицензия

[MIT](LICENSE). Используемые модели и библиотеки распространяются на условиях своих лицензий.
