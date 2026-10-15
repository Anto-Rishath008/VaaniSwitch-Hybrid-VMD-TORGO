[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$studyRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$accuracyPython = Join-Path $studyRoot '.venv-accuracy\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $accuracyPython)) { throw "Existing .venv-accuracy is missing: $accuracyPython" }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_XET = '1'
Push-Location -LiteralPath $studyRoot
try {
    & $accuracyPython -B (Join-Path $PSScriptRoot 'src\pipeline.py') run
    if ($LASTEXITCODE -ne 0) { throw "Accuracy study stopped (exit $LASTEXITCODE). Inspect experiments\accuracy_hubert\logs. Rerun the same command for safe pre-evaluation resume; an ambiguous outer claim requires review." }
} finally { Pop-Location }
