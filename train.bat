@echo off
REM ============================================================
REM  MEDIRAG-XAI — Model Training Script
REM  Run from project root: train.bat
REM ============================================================
cd /d "%~dp0backend"
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo.
echo  MEDIRAG-XAI — Training Disease Prediction Model
echo  Dataset: Kaggle Real Disease-Symptom (4,920 rows, 41 diseases)
echo.
python train_model.py
pause
