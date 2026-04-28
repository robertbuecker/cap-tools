$ErrorActionPreference = "Stop"

if (-not $args -or $args.Count -eq 0) {
    Write-Error "Usage: with-cap-tools-pip.ps1 <command> [args...]"
    exit 2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$condaExe = "C:\ProgramData\miniforge3\Scripts\conda.exe"

if (-not (Test-Path $condaExe)) {
    Write-Error "conda.exe not found at $condaExe"
    exit 1
}

# This shell starts with Restricted policy, which blocks the Conda PowerShell hook.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

$tmpDir = Join-Path $repoRoot ".tmp"
$numbaCacheDir = Join-Path $repoRoot ".numba_cache"
New-Item -ItemType Directory -Force -Path $tmpDir, $numbaCacheDir | Out-Null

$env:TEMP = $tmpDir
$env:TMP = $tmpDir
$env:NUMBA_CACHE_DIR = $numbaCacheDir

$hook = & $condaExe "shell.powershell" "hook" 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to initialize conda PowerShell hook: $hook"
    exit $LASTEXITCODE
}

$hook | Invoke-Expression
conda activate cap-tools-pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate conda environment 'cap-tools-pip'"
    exit $LASTEXITCODE
}

$commandName = $args[0]
$commandArgs = @()
if ($args.Count -gt 1) {
    $commandArgs = $args[1..($args.Count - 1)]
}

& $commandName @commandArgs
exit $LASTEXITCODE
