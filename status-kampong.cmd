@echo off
rem Is Kampong running, and is https://kampong.com.de reachable?
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\kampong.ps1" status
pause
