@echo off
setlocal
title Final SAC Dual-Delivery Robot
cd /d "%~dp0"
call "C:\Users\H ^& S\genesis_env\Scripts\activate.bat"
if errorlevel 1 (
  echo Could not activate Genesis environment.
  pause
  exit /b 1
)
python final_dual_delivery_robot_v5_deadlock_recovery.py
echo.
pause
endlocal
