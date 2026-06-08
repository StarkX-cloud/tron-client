# Run the TRON backend locally without Docker
# Usage: .\run_local.ps1
Set-StrictMode -Version Latest

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-Not (Test-Path .venv)) {
    python -m venv .venv
}

$activate = Join-Path .venv\Scripts\Activate.ps1
if (-Not (Test-Path $activate)) {
    Write-Error "Virtual environment not found at .venv\Scripts\Activate.ps1"
    exit 1
}

Write-Host "Activating venv..."
. $activate

Write-Host "Installing requirements if needed..."
pip install -r requirements.txt

Write-Host "Starting TRON queue server..."
python queue_server.py
