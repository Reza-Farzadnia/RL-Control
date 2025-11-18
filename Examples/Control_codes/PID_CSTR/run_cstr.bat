@echo off
REM Quick-start script for running CSTR Optimal Control

echo ============================================================
echo CSTR Optimal Control - Pyomo Implementation
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Install dependencies if needed
echo Checking dependencies...
pip show pyomo >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements_pyomo.txt
)

echo.
echo Running optimization...
echo ============================================================
echo.

python cstr_optimal_control.py

echo.
echo ============================================================
echo Execution complete!
echo ============================================================
pause
