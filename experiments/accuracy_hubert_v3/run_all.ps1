[CmdletBinding()]
param(
    [ValidateSet('Prepare','Validate','Develop','Outer')]
    [string]$Mode = 'Develop',
    [switch]$AuthorizeOuter
)
$ErrorActionPreference = 'Stop'
$StudyRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$PythonPath = Join-Path $StudyRoot '.venv-accuracy\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Existing .venv-accuracy is required.' }
if (($Mode -eq 'Outer') -and (-not $AuthorizeOuter)) {
    throw 'Real outer evaluation requires explicit -AuthorizeOuter.'
}
if ($AuthorizeOuter -and ($Mode -notin @('Develop','Outer'))) {
    throw '-AuthorizeOuter is valid only with Develop or Outer.'
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$Arguments = @('-B','-m','experiments.accuracy_hubert_v3.src.pipeline',$Mode.ToLowerInvariant())
if ($AuthorizeOuter) { $Arguments += '--authorize-outer' }
Push-Location -LiteralPath $StudyRoot
try {
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "V3 stopped with exit code $LASTEXITCODE; inspect V3 logs." }
} finally { Pop-Location }
