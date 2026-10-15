param(
    [ValidateSet('Prepare','Test','Validate','Develop','Outer','Preserve')]
    [string]$Mode = 'Validate',
    [switch]$AuthorizeDevelopment,
    [switch]$AuthorizeOuter
)
$ErrorActionPreference = 'Stop'
$V4Root = $PSScriptRoot
$StudyRoot = Split-Path (Split-Path $V4Root -Parent) -Parent
$Python = Join-Path $StudyRoot '.venv-accuracy\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Existing .venv-accuracy is missing.' }
if ($Mode -eq 'Develop' -and -not $AuthorizeDevelopment) { throw 'Full development requires separate approval and -AuthorizeDevelopment.' }
if ($Mode -eq 'Outer' -and -not $AuthorizeOuter) { throw 'Outer requires separate explicit authorization and -AuthorizeOuter.' }
if ($AuthorizeDevelopment -and $Mode -ne 'Develop') { throw 'Development flag only applies to Develop.' }
if ($AuthorizeOuter -and $Mode -ne 'Outer') { throw 'Outer flag only applies to a separate Outer invocation.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MPLCONFIGDIR = Join-Path $V4Root 'cache\matplotlib'
$LogRoot = Join-Path $V4Root 'logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$LogPath = Join-Path $LogRoot ("launcher-{0}-{1}.log" -f $Mode, [guid]::NewGuid().ToString('N'))
Push-Location $StudyRoot
try {
    Start-Transcript -LiteralPath $LogPath | Out-Null
    if ($Mode -in @('Test','Validate','Develop')) {
        & $Python -B -m unittest discover -s experiments/accuracy_wavlm_v4/tests -v
        if ($LASTEXITCODE -ne 0) { throw 'V4 automated tests failed.' }
    }
    if ($Mode -ne 'Test') {
        $Arguments = @('-B','-m','experiments.accuracy_wavlm_v4.src.pipeline','--mode',$Mode)
        if ($AuthorizeDevelopment) { $Arguments += '--authorize-development' }
        if ($AuthorizeOuter) { $Arguments += '--authorize-outer' }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "V4 $Mode failed." }
    }
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    Pop-Location
}
