param(
    [Parameter(Mandatory=$true)][string]$Freeze,
    [Parameter(Mandatory=$true)][string]$FreezeSha256
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = (Join-Path $Root "src")
& ".\.venv-m6\Scripts\python.exe" -B "scripts\module6_phase234.py" final-eval --freeze $Freeze --freeze-sha256 $FreezeSha256
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
