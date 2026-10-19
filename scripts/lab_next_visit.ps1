[CmdletBinding()]
param(
    [ValidateSet('Check', 'Initialize', 'Diagnostics', 'Events', 'Capture', 'Observe', 'Release', 'Empty', 'Round', 'Train')]
    [string]$Action = 'Check',
    [string]$Profile = 'config/lab_next_visit.json',
    [string]$Python = '',
    [string]$Release = '',
    [string]$Session = '',
    [switch]$Live
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $repo '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python missing. Pass -Python with the lab CPython 3.12 executable.'
}
$launcherArgs = @('-X', 'utf8', '-m', 'scripts.lab_session', $Action.ToLowerInvariant(), '--profile', $Profile)
if ($Live) { $launcherArgs += '--live' }
if ($Release) { $launcherArgs += @('--empty-release', $Release) }
if ($Session) { $launcherArgs += @('--session', $Session) }
Push-Location -LiteralPath $repo
try {
    & $Python @launcherArgs
    if ($LASTEXITCODE -ne 0) { throw 'Blocked or failed. Read the saved session_status.json and logs; do not automatically retry.' }
} finally {
    Pop-Location
}
