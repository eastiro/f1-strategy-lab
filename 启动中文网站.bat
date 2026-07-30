@echo off
setlocal
title F1 Strategy Lab

cd /d "%~dp0"

echo.
echo ========================================
echo   F1 Strategy Lab
echo   URL: http://localhost:8506
echo ========================================
echo.
echo Starting the Chinese research site...

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8506/'"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_streamlit.ps1" -Port 8506

if errorlevel 1 (
    echo.
    echo Startup failed. Keep this window open and check the error above.
    pause
)

endlocal
