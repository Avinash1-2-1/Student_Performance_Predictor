@echo off
echo ============================================
echo  Installing Student Performance Predictor
echo ============================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found on your system.
    echo Please install Python from https://www.python.org/downloads/
    echo IMPORTANT: during install, check the box "Add python.exe to PATH"
    pause
    exit /b
)

echo Installing required packages, this may take a minute...
pip install -r requirements.txt

echo.
echo ============================================
echo  Done! You can now double-click run_app.bat
echo ============================================
pause
