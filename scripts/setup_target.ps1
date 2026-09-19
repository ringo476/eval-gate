# Vendors the target app the harness evaluates, pinned to an exact commit.
$ErrorActionPreference = "Stop"
$TargetDir = "targets/rag-fusion"
$TargetRepo = "https://github.com/Raudaschl/rag-fusion.git"
$TargetCommit = "e6ae2d44f440cd52356c18ebef2b8bcb5cb436b9"

if (Test-Path "$TargetDir/.git") {
    Write-Host "Target already present at $TargetDir"
} else {
    Write-Host "Cloning $TargetRepo ..."
    git clone --quiet $TargetRepo $TargetDir
}

Push-Location $TargetDir
git fetch --quiet origin $TargetCommit 2>$null
git checkout --quiet $TargetCommit
Pop-Location
Write-Host "Pinned targets/rag-fusion to $TargetCommit"
