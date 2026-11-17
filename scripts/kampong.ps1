# Start, stop or check Kampong as a website hosted on this computer.
#   Kampong itself: the Python server on port 8700, listening on this computer only.
#   The Cloudflare Tunnel "kampong": passes visitors from https://kampong.com.de to that server.
# Usually run through start-kampong.cmd / stop-kampong.cmd in the project folder.
param([ValidateSet("start", "stop", "status")] [string]$Action = "status")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$Port = 8700
$Site = "https://kampong.com.de"
$Logs = Join-Path $Root "logs"

function Server-Pid { (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess }
function Tunnel-Procs { Get-Process cloudflared -ErrorAction SilentlyContinue }
function Local-Ok { try { (Invoke-WebRequest "http://127.0.0.1:$Port/api/me" -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 } catch { $false } }
function Site-Status { try { (Invoke-WebRequest "$Site/api/me" -UseBasicParsing -TimeoutSec 6).StatusCode } catch { if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 } } }

function Show-Status {
    $p = Server-Pid
    Write-Host ("Kampong server : " + $(if ($p) { "running (process $p, port $Port)" } else { "stopped" }))
    Write-Host ("Tunnel         : " + $(if (Tunnel-Procs) { "running" } else { "stopped" }))
    if (Tunnel-Procs) {
        $code = Site-Status
        Write-Host ("$Site : " + $(if ($code -eq 200) { "online" } elseif ($code -eq 530) { "not reachable yet (the tunnel isn't connected; see logs\tunnel.log)" } else { "answered $code" }))
    }
}

if ($Action -eq "status") { Show-Status; exit 0 }

if ($Action -eq "stop") {
    if (Tunnel-Procs) { Tunnel-Procs | Stop-Process -Force; Write-Host "Tunnel stopped." } else { Write-Host "Tunnel was not running." }
    $p = Server-Pid
    if ($p) { Stop-Process -Id $p -Force; Write-Host "Kampong server stopped." } else { Write-Host "Kampong server was not running." }
    Write-Host "Kampong is offline."
    exit 0
}

# ---- start
if (-not (Test-Path $Python)) { Write-Host "Can't find the project's Python at $Python." -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Cloudflared)) { Write-Host "Can't find Cloudflare's connector at $Cloudflared." -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force $Logs | Out-Null

if (Server-Pid) {
    Write-Host "Kampong server is already running on port $Port; using it."
} else {
    Write-Host "Starting Kampong..."
    # --proxy-headers: trust what Cloudflare's connector (on this computer) says about https and the visitor
    Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Minimized `
        -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$Port", "--proxy-headers" `
        -RedirectStandardOutput (Join-Path $Logs "kampong.log") -RedirectStandardError (Join-Path $Logs "kampong-errors.log")
    $ok = $false
    foreach ($i in 1..60) { Start-Sleep -Seconds 1; if (Local-Ok) { $ok = $true; break } }
    if (-not $ok) { Write-Host "Kampong didn't start within a minute. See logs\kampong-errors.log." -ForegroundColor Red; exit 1 }
    Write-Host "Kampong is running on this computer."
}

if (Tunnel-Procs) {
    Write-Host "Tunnel is already running."
} else {
    Write-Host "Connecting to Cloudflare..."
    Start-Process -FilePath $Cloudflared -ArgumentList "tunnel", "run", "kampong" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Logs "tunnel-out.log") -RedirectStandardError (Join-Path $Logs "tunnel.log")
}

$code = 0
Write-Host "Checking $Site (up to about a minute)..."
foreach ($i in 1..10) { Start-Sleep -Seconds 3; $code = Site-Status; if ($code -eq 200) { break } }
if ($code -eq 200) {
    Write-Host ""
    Write-Host "Kampong is online at $Site" -ForegroundColor Green
    Write-Host "Keep this computer on, awake and online. Run stop-kampong.cmd to take it offline."
} else {
    Write-Host ""
    Write-Host "Kampong is running here, but $Site isn't reachable yet (answer: $code)." -ForegroundColor Yellow
    Write-Host "If logs\tunnel.log mentions port 7844, this network blocks Cloudflare Tunnel: try home Wi-Fi or a phone hotspot."
}
