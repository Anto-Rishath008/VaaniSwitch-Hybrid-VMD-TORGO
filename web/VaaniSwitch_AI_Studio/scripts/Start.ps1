$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $studioRoot
$studioPython = Join-Path $studioRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $studioPython)) { throw 'Run scripts\Setup.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $studioRoot 'frontend\dist\index.html'))) { throw 'Run scripts\Setup.ps1 to build the frontend.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Host 'VaaniSwitch AI Studio: http://127.0.0.1:8000'
Write-Host 'Backend: http://127.0.0.1:8000/api/health'
Write-Host 'OpenAPI: http://127.0.0.1:8000/openapi.json'
Write-Host 'Press Ctrl+C to stop and remove temporary audio sessions.'
& $studioPython -B -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
if ($LASTEXITCODE -ne 0) { throw 'The local server stopped with an error. Check whether port 8000 is already in use.' }
