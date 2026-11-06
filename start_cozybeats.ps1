$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Test-LocalHttp($url) {
  try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true }
  catch { return $false }
}

if (-not (Test-LocalHttp "http://127.0.0.1:8000/healthz")) {
  Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "Set-Location '$root'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload *> api.log") | Out-Null
}
if (-not (Test-LocalHttp "http://127.0.0.1:8501")) {
  Start-Process powershell -WindowStyle Normal -WorkingDirectory $root -ArgumentList @("-NoProfile", "-Command", "python -m streamlit run app.py --server.port 8501") | Out-Null
}
if (-not (Test-LocalHttp "http://127.0.0.1:3000")) {
  if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) { npm install --prefix (Join-Path $root "frontend") }
  Start-Process powershell -WindowStyle Normal -WorkingDirectory (Join-Path $root "frontend") -ArgumentList @("-NoProfile", "-Command", "`$env:NODE_OPTIONS='--max-old-space-size=4096'; npm run dev") | Out-Null
}
Start-Sleep -Seconds 3
Start-Process "http://localhost:3000" | Out-Null
Write-Host "CozyBeats started: frontend 3000, API 8000, Streamlit 8501"
