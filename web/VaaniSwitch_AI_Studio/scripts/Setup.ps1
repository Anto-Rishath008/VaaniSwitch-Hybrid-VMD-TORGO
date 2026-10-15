$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $studioRoot
$projectParent = Split-Path -Parent $studioRoot
$accuracyPackages = Join-Path $projectParent 'VaaniSwitch_Accuracy_Study\.venv-accuracy\Lib\site-packages'
$speechPackages = Join-Path $projectParent 'VaaniSwitch_FINAL_Hybrid_VMD_TORGO_Project\.venv-m6\Lib\site-packages'
foreach ($requiredPath in @($accuracyPackages,$speechPackages)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Required read-only research runtime is missing: $requiredPath" }
}
$studioPython = Join-Path $studioRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $studioPython)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating the application environment failed.' }
}
Set-Content -LiteralPath (Join-Path $studioRoot '.venv\Lib\site-packages\research_readonly.pth') -Value @($accuracyPackages,$speechPackages)
& $studioPython -m pip install -r requirements-web.lock
if ($LASTEXITCODE -ne 0) { throw 'Installing the application web dependencies failed.' }
# The reused environments are read-only. pywin32 needs local DLL/bootstrap paths.
& $studioPython -B -c 'import pywintypes, pythoncom'
if ($LASTEXITCODE -ne 0) {
    & $studioPython -m pip install --ignore-installed --no-deps pywin32==312
    if ($LASTEXITCODE -ne 0) { throw 'Installing local Windows speech bindings failed.' }
}
Push-Location -LiteralPath (Join-Path $studioRoot 'frontend')
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup complete. Run: .\scripts\Start.ps1'
