@echo off
REM ============================================================
REM  MEDIRAG-XAI — Automated Windows Setup Script
REM  Run this once to set up the entire project
REM ============================================================

echo.
echo  =============================================
echo    MEDIRAG-XAI — Healthcare AI Platform Setup
echo  =============================================
echo.

cd /d "%~dp0backend"

echo [1/6] Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

echo [2/6] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/6] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Package installation failed. Check requirements.txt
    pause & exit /b 1
)

echo [5/6] Downloading spaCy language model...
python -m spacy download en_core_web_sm
echo (SciSpacy model optional - install manually if needed)

echo [6/6] Training disease prediction model...
echo NOTE: This uses the real Kaggle dataset (Training.csv / Testing.csv)
echo       4,920 training samples, 41 diseases, 132 symptoms
python train_model.py

echo.
echo  =============================================
echo    Setup Complete!
echo  =============================================
echo.
echo  To start the backend:
echo    cd MediRAG-XAI\backend
echo    .venv\Scripts\activate
echo    python main.py
echo.
echo  To view the frontend:
echo    Open frontend\index.html in a browser
echo    Or: python -m http.server 5500
echo.
echo  Backend API docs: http://localhost:8000/docs
echo.
pause
