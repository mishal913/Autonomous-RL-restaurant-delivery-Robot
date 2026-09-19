@echo off
setlocal

title Final SAC Dual-Delivery Robot

REM ==========================================================
REM ALWAYS RUN FROM THIS BAT FILE'S FOLDER
REM ==========================================================
cd /d "%~dp0"

echo.
echo ============================================================
echo   FINAL SAC DUAL-DELIVERY ROBOT
echo ============================================================
echo.
echo Activating Genesis environment...

REM Your working Python/Genesis environment from the successful run.
call "C:\Users\H & S\genesis_env\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate:
    echo C:\Users\H ^& S\genesis_env
    echo.
    pause
    exit /b 1
)

echo Environment activated.
echo Starting final project...
echo.

python final_dual_delivery_robot_v5_deadlock_recovery.py

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
    echo Project finished normally.
) else (
    echo Project stopped with exit code %EXIT_CODE%.
)
echo ============================================================
echo.
pause

endlocal
