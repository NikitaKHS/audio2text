@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found. Install Python 3.10 or newer.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating local virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)

echo Installing verified dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 exit /b 1

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA GPU detected. Installing CUDA 12 and cuDNN 9 runtime...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-gpu.txt
  if errorlevel 1 exit /b 1
)

echo.
echo audio2text is ready.
exit /b 0
