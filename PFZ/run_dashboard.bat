@echo off
title AI-Based Dynamic Fishing Zone Prediction System
cd /d "%~dp0"
echo ========================================================
echo   Launching AI-Based Dynamic Fishing Zone Dashboard...
echo ========================================================
echo.

if exist "%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe" (
    "%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe" -m streamlit run code_base/dashboard.py
    goto end
)

if exist "%LOCALAPPDATA%\Python\bin\python3.11.exe" (
    "%LOCALAPPDATA%\Python\bin\python3.11.exe" -m streamlit run code_base/dashboard.py
    goto end
)

python -m streamlit run code_base/dashboard.py

:end
pause
