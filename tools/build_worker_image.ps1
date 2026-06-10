# Build and push the vw-studio-worker image used by remote HF Jobs stages.
# Requires Docker Desktop and a Docker Hub login (docker login).
#
# Usage:
#   .\tools\build_worker_image.ps1 -Repo myuser/vw-studio-worker -Tag 0.1
#   .\tools\build_worker_image.ps1 -Repo myuser/vw-studio-worker -Tag 0.1 -Push

param(
    [Parameter(Mandatory = $true)] [string]$Repo,
    [string]$Tag = "0.1",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$context = Join-Path "C:\Users\Administrator\Desktop\Github Repos\vaultwares-studio" "docker\worker"
$image = "${Repo}:${Tag}"

Write-Host "Building $image from $context"
docker build -t $image $context
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

if ($Push) {
    Write-Host "Pushing $image"
    docker push $image
    if ($LASTEXITCODE -ne 0) { throw "docker push failed" }
    Write-Host "Done. Set worker_image=$image in data/remote_compute.json (or Settings)."
} else {
    Write-Host "Built $image (use -Push to publish to Docker Hub)."
}
