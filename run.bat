@echo off
REM ============================================================
REM  MEDIRAG-XAI — Quick Start Server
REM  Run from project root: run.bat
REM ============================================================
cd /d "%~dp0backend"
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo.
echo  Starting MEDIRAG-XAI Backend...
echo  API:  http://localhost:8000
echo  Docs: http://localhost:8000/docs
echo.
python main.py
