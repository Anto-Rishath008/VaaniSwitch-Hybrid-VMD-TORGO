$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = (Join-Path $Root "src")
& ".\.venv-m6\Scripts\python.exe" -B "scripts\module6_phase234.py" pre-eval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
