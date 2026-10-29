@echo off
setlocal
cd /d "%~dp0"
title Vocabulary Learning Assistant

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:5000"
set "LOG=%ROOT%startup.log"

if not exist "%PYTHON%" (
  echo First run: creating Python environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv "%ROOT%.venv"
  ) else (
    python -m venv "%ROOT%.venv"
  )
  if errorlevel 1 goto PYTHON_ERROR
)

"%PYTHON%" -c "import flask, openai, httpx2, dotenv, reportlab" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages...
  "%PYTHON%" -m pip install -r "%ROOT%requirements.txt"
  if errorlevel 1 goto INSTALL_ERROR
)

if not exist "%ROOT%.env" (
  copy /y "%ROOT%.env.example" "%ROOT%.env" >nul
  echo A local .env file has been created.
  echo Please edit it and add your API configuration, then run this file again.
  pause
  exit /b 1
)

start "Vocabulary Learning Assistant" /min "%PYTHON%" "%ROOT%app.py" > "%LOG%" 2>&1

set /a COUNT=0
:WAIT
set /a COUNT+=1
curl.exe -s "%URL%/api/health" >nul 2>nul
if not errorlevel 1 goto OPEN
if %COUNT% GEQ 30 goto SERVER_ERROR
timeout /t 1 /nobreak >nul
goto WAIT

:OPEN
start "" "%URL%"
exit /b 0

:PYTHON_ERROR
echo Python 3.10 or newer is required.
pause
exit /b 1

:INSTALL_ERROR
echo Failed to install dependencies. Check the network and try again.
pause
exit /b 1

:SERVER_ERROR
echo The service did not start. See startup.log for details.
if exist "%LOG%" type "%LOG%"
pause
exit /b 1
