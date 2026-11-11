# Registers the weekly Oikos data refresh with Windows Task Scheduler (current user, no admin needed).
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_refresh.ps1            set it up (Sundays 03:00)
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_refresh.ps1 -Remove    take it away
# If the laptop is off or asleep at 03:00, the refresh runs the next time you are logged in.
param([switch]$Remove)
$name = "Oikos data refresh"
if ($Remove) { Unregister-ScheduledTask -TaskName $name -Confirm:$false; "Removed."; return }
$bat = Join-Path (Split-Path $PSScriptRoot -Parent) "scripts\refresh.bat"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 8) -Priority 4 -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Description "Downloads the latest HDB resale data and retrains the Oikos models." -Force | Out-Null
Get-ScheduledTask -TaskName $name | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime, LastRunTime, LastTaskResult
