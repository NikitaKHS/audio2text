@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите setup.bat
    pause
    exit /b 1
)

echo Установка высокоточного русского режима GigaAM-v3...
".venv\Scripts\python.exe" -m pip install -r requirements-accuracy.txt
if errorlevel 1 (
    echo Ошибка установки.
    pause
    exit /b 1
)

echo Готово. Пример запуска:
echo .venv\Scripts\python.exe transcribe_accuracy.py 1.m4a 2.m4a
pause
