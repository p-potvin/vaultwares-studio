$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Python = "python.exe"

Write-Host "Installing PyInstaller into the local virtual environment..." -ForegroundColor Cyan
& $Python -m pip install pyinstaller

Write-Host "Building usd-playground-demo.exe..." -ForegroundColor Cyan
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "usd-playground-demo" `
    --paths "$RepoRoot" `
    --collect-submodules "pxr" `
    --collect-binaries "pxr" `
    --collect-data "pxr" `
    --add-data "test_input.mp4;." `
    demo_launcher.py

Write-Host "Build complete: dist\usd-playground-demo.exe" -ForegroundColor Green
