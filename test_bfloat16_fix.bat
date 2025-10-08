@echo off
echo ====================================
echo bfloat16 Fix Verification Test
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
echo Running bfloat16 vs float32 test...
echo.
python test_bfloat16_fix.py

if errorlevel 1 (
    echo.
    echo ERROR: Test failed
    pause
    exit /b 1
)

pause

