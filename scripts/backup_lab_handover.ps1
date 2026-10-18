param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$target = [System.IO.Path]::GetFullPath((Join-Path $repo $Destination))
$backupRoot = [System.IO.Path]::GetFullPath((Join-Path $repo "backups"))
if (-not ($target.StartsWith($backupRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase))) {
    throw "Destination must be a new child of $backupRoot"
}
if (Test-Path -LiteralPath $target) {
    throw "Refusing to overwrite existing backup: $target"
}

New-Item -ItemType Directory -Path $target | Out-Null

function Copy-RepoItem {
    param([string]$RelativePath)
    $source = Join-Path $repo $RelativePath
    if (-not (Test-Path -LiteralPath $source)) {
        return $false
    }
    $destination = Join-Path $target $RelativePath
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse
    return $true
}

$copied = [ordered]@{}
$copied["results/lab_visit"] = Copy-RepoItem "results/lab_visit"
$copied["results/robot_leg_round"] = Copy-RepoItem "results/robot_leg_round"
$copied["results/reference_commissioning"] = Copy-RepoItem "results/reference_commissioning"
$copied["config"] = Copy-RepoItem "config"
$copied["site_evidence"] = Copy-RepoItem "site_evidence"
$copied["docs/NEXT_LAB_VISIT.md"] = Copy-RepoItem "docs/NEXT_LAB_VISIT.md"
$copied["_rokae_log_"] = Copy-RepoItem "_rokae_log_"

$diagnosticTarget = Join-Path $target "diagnostics"
New-Item -ItemType Directory -Path $diagnosticTarget -Force | Out-Null
$diagnosticFiles = Get-ChildItem -LiteralPath (Join-Path $repo "diagnostics") -File |
    Where-Object { $_.Name -like "*20261018*" }
foreach ($file in $diagnosticFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $diagnosticTarget $file.Name)
}
$copied["diagnostics/*20261018*"] = $diagnosticFiles.Count -gt 0

$photoRoots = @("photos", "site_photos", "site_evidence/photos")
$photoCount = 0
foreach ($relative in $photoRoots) {
    $source = Join-Path $repo $relative
    if (Test-Path -LiteralPath $source) {
        $destination = Join-Path $target $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Recurse
        $photoCount += @(Get-ChildItem -LiteralPath $source -Recurse -File).Count
    }
}

$missingLines = @()
if (-not $copied["results/robot_leg_round"]) {
    $missingLines += "results/robot_leg_round was not created because no robot motion round ran."
}
if ($photoCount -eq 0) {
    $missingLines += "No site photo directory/files were present in the workspace at backup time."
}
[System.IO.File]::WriteAllLines((Join-Path $target "MISSING_ITEMS.txt"), $missingLines, [System.Text.UTF8Encoding]::new($false))

$files = Get-ChildItem -LiteralPath $target -Recurse -File | Where-Object { $_.Name -notin @("manifest.json", "SHA256SUMS.txt") }
$entries = foreach ($file in $files) {
    $relative = $file.FullName.Substring($target.Length).TrimStart('\', '/').Replace('\', '/')
    [pscustomobject][ordered]@{
        path = $relative
        size_bytes = $file.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString("o")
    source_repo = $repo
    destination = $target
    copied_sections = $copied
    photo_file_count = $photoCount
    file_count = @($entries).Count
    total_bytes = ($entries | Measure-Object -Property size_bytes -Sum).Sum
    files = @($entries)
}
[System.IO.File]::WriteAllText(
    (Join-Path $target "manifest.json"),
    ($manifest | ConvertTo-Json -Depth 8),
    [System.Text.UTF8Encoding]::new($false)
)
$sumLines = $entries | Sort-Object path | ForEach-Object { "$($_.sha256)  $($_.path)" }
[System.IO.File]::WriteAllLines((Join-Path $target "SHA256SUMS.txt"), $sumLines, [System.Text.UTF8Encoding]::new($false))

$csvMismatches = @()
$sourceCsvRoot = Join-Path $repo "results/lab_visit"
$sourceCsv = Get-ChildItem -LiteralPath $sourceCsvRoot -Recurse -File -Filter "*.csv"
foreach ($source in $sourceCsv) {
    $relative = $source.FullName.Substring($sourceCsvRoot.Length).TrimStart('\', '/')
    $copy = Join-Path (Join-Path $target "results/lab_visit") $relative
    if (-not (Test-Path -LiteralPath $copy)) {
        $csvMismatches += "missing:$relative"
        continue
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source.FullName).Hash
    $copyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $copy).Hash
    if ($sourceHash -ne $copyHash) {
        $csvMismatches += "hash:$relative"
    }
}
if ($csvMismatches.Count -gt 0) {
    throw "CSV backup verification failed: $($csvMismatches -join ', ')"
}

[ordered]@{
    backup_path = $target
    file_count = @($entries).Count
    total_bytes = $manifest.total_bytes
    lab_visit_csv_count = $sourceCsv.Count
    lab_visit_csv_hash_mismatches = 0
    robot_leg_round_present = [bool]$copied["results/robot_leg_round"]
    photo_file_count = $photoCount
} | ConvertTo-Json -Depth 4
