@echo off
rem Weekly data refresh for Oikos: downloads the latest HDB sales, retrains, and rebuilds the estimates.
rem Run by the Windows scheduled task "Oikos data refresh" (see scripts\schedule_refresh.ps1). Output goes to data\update.log.
rem The running server notices the new models by itself; no restart is needed.
cd /d "%~dp0.."
echo. >> data\update.log
echo ==== refresh started %date% %time% ==== >> data\update.log
".venv\Scripts\python.exe" -u src\update.py >> data\update.log 2>&1
echo ==== refresh ended %date% %time% with exit code %errorlevel% ==== >> data\update.log
