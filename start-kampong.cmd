@echo off
rem Put Kampong online at https://kampong.com.de from this computer (Kampong server + Cloudflare Tunnel).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\kampong.ps1" start
pause
