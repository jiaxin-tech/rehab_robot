[CmdletBinding()]
param(
    [ValidateSet('Check', 'Initialize', 'Diagnostics', 'Events', 'Capture', 'Observe', 'Release', 'Empty', 'Round', 'Train')]
    [string]$Action = 'Check',
    [string]$Python = '',
    [string]$Release = '',
    [string]$Session = '',
    [switch]$Live
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$parameters = @{Action=$Action; Profile=(Join-Path $PSScriptRoot 'launcher.json')}
if ($Python) { $parameters.Python = $Python }
if ($Release) { $parameters.Release = $Release }
if ($Session) { $parameters.Session = $Session }
if ($Live) { $parameters.Live = $true }
& (Join-Path $repo 'scripts/lab_next_visit.ps1') @parameters
