@echo off
echo ====================================
echo LoRA Performance Diagnostic Test
echo ====================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate venv
    echo Make sure venv exists in the current directory
    pause
    exit /b 1
)

echo.
echo Running performance test...
echo.
python test_lora_performance.py

if errorlevel 1 (
    echo.
    echo ERROR: Test failed to run
    pause
    exit /b 1
)

echo.
echo Test completed successfully!
pause

