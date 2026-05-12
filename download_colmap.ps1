# download_colmap.ps1
# This script downloads a portable version of COLMAP for Windows.

$url = "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip"
$dest = "colmap.zip"
$extractPath = "$PSScriptRoot\tools\colmap"

Write-Host "Downloading COLMAP..."
Invoke-WebRequest -Uri $url -OutFile $dest

Write-Host "Extracting COLMAP to tools'colmap..."
if (-not (Test-Path $extractPath)) { New-Item -ItemType Directory -Path $extractPath }
Expand-Archive -Path $dest -DestinationPath $extractPath -Force

Write-Host "Cleaning up..."
Remove-Item $dest

Write-Host "COLMAP installed to $extractPath"
Write-Host "Please add $($extractPath)\bin to your system PATH to enable the ReconstructionAgent."
