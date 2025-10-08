@echo off
echo ====================================
echo PyTorch Installation Deep Check
echo ====================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate venv
    pause
    exit /b 1
)

echo.
echo Running deep diagnostic check...
echo.
python check_pytorch_install.py

if errorlevel 1 (
    echo.
    echo ERROR: Test failed
    pause
    exit /b 1
)

pause

