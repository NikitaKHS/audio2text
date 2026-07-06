@echo off
cd /d "%~dp0"
py -3 -m pip install -q customtkinter 2>nul
py -3 transcribe_gui.py
if errorlevel 1 pause
