@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
if not exist .venv\.agora-installed (
  .venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-lock.txt
  if errorlevel 1 goto failed
  type nul > .venv\.agora-installed
)
.venv\Scripts\python.exe run.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Agora could not start. See the error above and docs\GETTING_STARTED.md.
pause
exit /b 1
