[CmdletBinding()]
param(
    [ValidateSet('Check', 'Initialize', 'Diagnostics', 'Events', 'Capture', 'Quality', 'Calibrate', 'Observe', 'Release', 'Empty', 'Round', 'Train')]
    [string]$Action = 'Check',
    [string]$Profile = 'config/lab_next_visit.json',
    [string]$Python = '',
    [string]$Release = '',
    [string]$Session = '',
    [string]$OutputRoot = '',
    [string]$Data = '',
    [string]$Manifest = '',
    [string]$Label = '',
    [ValidateRange(0.1, 3600.0)]
    [double]$DurationSeconds = 10,
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
if ($OutputRoot) { $launcherArgs += @('--output-root', $OutputRoot) }
if ($Data) { $launcherArgs += @('--data', $Data) }
if ($Manifest) { $launcherArgs += @('--manifest', $Manifest) }
if ($Label) { $launcherArgs += @('--label', $Label) }
if ($PSBoundParameters.ContainsKey('DurationSeconds')) { $launcherArgs += @('--duration-s', $DurationSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
Push-Location -LiteralPath $repo
try {
    & $Python @launcherArgs
    if ($LASTEXITCODE -ne 0) { throw 'Blocked or failed. Read the saved session_status.json and logs; do not automatically retry.' }
} finally {
    Pop-Location
}
