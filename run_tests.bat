@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup.bat
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -c "from importlib.metadata import version; assert (version('faster-whisper'), version('tqdm'), version('customtkinter')) == ('1.2.1', '4.69.0', '6.0.0')" >nul 2>nul
if errorlevel 1 call setup.bat
if errorlevel 1 goto :error

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  ".venv\Scripts\python.exe" -c "from importlib.metadata import version; assert (version('nvidia-cublas-cu12'), version('nvidia-cuda-nvrtc-cu12'), version('nvidia-cudnn-cu12')) == ('12.9.2.10', '12.9.86', '9.24.0.43')" >nul 2>nul
  if errorlevel 1 call setup.bat
  if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :test_error
exit /b 0

:test_error
echo Tests failed.
pause
exit /b 1

:error
echo Failed to prepare the test environment.
pause
exit /b 1
