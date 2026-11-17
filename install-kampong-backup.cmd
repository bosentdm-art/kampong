@echo off
rem Sets up the nightly backup of Kampong's database to Cloudflare R2 (Windows task "Kampong backup", 3am daily).
rem To remove it later: Task Scheduler > Task Scheduler Library > "Kampong backup" > Delete.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\kampong.ps1" install-backup
pause
