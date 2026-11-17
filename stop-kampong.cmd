@echo off
rem Take Kampong offline: stops the Cloudflare Tunnel and the Kampong server on this computer.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\kampong.ps1" stop
pause
