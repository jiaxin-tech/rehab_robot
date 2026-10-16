[CmdletBinding()]
param(
    [ValidateSet('Check', 'Capture', 'Quality', 'Calibrate', 'Prepare', 'Run')]
    [string]$Action = 'Check',
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'config/robot_leg_round.json',
    [string]$LegConfig = 'config/twincat_leg_capture.json',
    [string]$OutputRoot = 'results/lab_visit',
    [string]$Session = '',
    [string]$Data = '',
    [string]$Manifest = '',
    [string]$Label = 'static',
    [ValidateRange(0.1, 3600.0)]
    [double]$DurationSeconds = 10,
    [string]$Confirmation = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$python = Join-Path $Repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python missing: $python. Restore the reviewed project environment first."
}
if ($Label -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'Label must contain only ASCII letters, numbers, underscores and hyphens.'
}
Push-Location -LiteralPath $Repo
try {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $output = [System.IO.Path]::GetFullPath($OutputRoot)
    [System.IO.Directory]::CreateDirectory($output) | Out-Null
    $code = 0
    switch ($Action) {
        'Check' {
            & $python -X utf8 -c 'import sys,numpy,pandas,scipy,mujoco; print("Python:",sys.version); print("Offline dependencies imported; no device connection.")'
            if ($LASTEXITCODE -ne 0) { throw 'Offline dependency check failed.' }
            $report = Join-Path $output "readiness_$stamp.json"
            & $python -X utf8 -m scripts.run_robot_leg_round check --config $Config --output $report
            $code = $LASTEXITCODE
            Write-Host "Readiness report: $report"
            if ($code -eq 1) { Write-Host 'Configuration has blockers. Complete the checklist before motion.' }
        }
        'Capture' {
            $segment = Join-Path $output "${Label}_$stamp"
            Write-Host "Read-only ADS capture; no enable, reset or motion: $segment"
            & $python -X utf8 -m scripts.capture_leg_segment --config $LegConfig --output-dir $segment --duration-s $DurationSeconds
            $code = $LASTEXITCODE
        }
        'Quality' {
            if (-not $Data) { throw '-Data must name one raw CSV or a directory containing raw CSV files.' }
            & $python -X utf8 -m measurement_validation.offline_leg quality $Data --output-dir (Join-Path $output "quality_$stamp")
            $code = $LASTEXITCODE
        }
        'Calibrate' {
            if (-not $Data -or -not $Manifest) { throw '-Data and -Manifest are required. Use actual measured angles.' }
            & $python -X utf8 -m measurement_validation.offline_leg calibrate $Manifest --data-dir $Data --output (Join-Path $output "calibration_$stamp.json")
            $code = $LASTEXITCODE
        }
        'Prepare' {
            if (-not $Session) { throw '-Session must name a new round directory.' }
            Write-Host 'Read robot pose and prepare preview. No robot motion is commanded.'
            & $python -X utf8 -m scripts.run_robot_leg_round prepare --config $Config --session $Session
            $code = $LASTEXITCODE
        }
        'Run' {
            if (-not $Session) { throw '-Session must name the previously prepared round directory.' }
            if ($Confirmation -cne 'I CONFIRM REVIEWED ROBOT LEG ROUND') {
                throw 'Actual motion requires the existing exact on-site confirmation via -Confirmation.'
            }
            Write-Host 'ACTUAL ROBOT MOTION: executing one explicitly confirmed round.'
            & $python -X utf8 -m scripts.run_robot_leg_round run --session $Session --confirmation $Confirmation
            $code = $LASTEXITCODE
        }
    }
    if ($code -ne 0) { throw "Action $Action failed or was blocked (exit $code). Keep outputs and inspect the report." }
} finally {
    Pop-Location
}
