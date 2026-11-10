$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "IAQ RESEARCH PIPELINE V2 - WINDOWS PREPARATION"
Write-Host "============================================================"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProjectRoot "config\windows.json"
$RawDataDir = Join-Path $ProjectRoot "data\raw"
$OutputDirs = @(
    "outputs\processed",
    "outputs\models",
    "outputs\metrics",
    "outputs\figures",
    "outputs\logs"
)

if (-not (Test-Path $ConfigPath)) {
    throw "Configuration file not found: $ConfigPath"
}

New-Item -ItemType Directory -Force -Path $RawDataDir | Out-Null
foreach ($RelativeDir in $OutputDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $RelativeDir) | Out-Null
}

$RequiredFiles = @("zone_co2.csv", "zone_temp_interior.csv")
$MissingFiles = @()
foreach ($FileName in $RequiredFiles) {
    $FilePath = Join-Path $RawDataDir $FileName
    if (Test-Path $FilePath) {
        $Item = Get-Item $FilePath
        Write-Host "[FOUND] $FileName ($([math]::Round($Item.Length / 1MB, 2)) MB)"
    }
    else {
        $MissingFiles += $FileName
        Write-Host "[MISSING] $FileName"
    }
}

Write-Host ""
Write-Host "Project root: $ProjectRoot"
Write-Host "Python environment expected: ML_CLASS"
Write-Host "No packages have been installed or upgraded by this script."

if ($MissingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Copy these two original files into:"
    Write-Host $RawDataDir
    foreach ($FileName in $MissingFiles) {
        Write-Host " - $FileName"
    }
    Write-Host ""
    Write-Host "Do not put the complete Building 59 folder here."
    exit 2
}

Write-Host ""
Write-Host "[OK] Both source CSV files are in place."
Write-Host "[OK] Output directories are ready."
Write-Host ""
Write-Host "Next: activate ML_CLASS and run the environment check described in README.md."
