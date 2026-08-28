param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

$ErrorActionPreference = "Stop"

Write-Host "=== Agent-UniRAG Research Git Sync ===" -ForegroundColor Cyan

# Never stage the contents of the reference submodule.
git add --all -- . ':(exclude)original/Agent-UniRAG'

if ($LASTEXITCODE -ne 0) {
    throw "git add failed."
}

git diff --cached --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "No parent-repository changes to commit." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStaged changes:" -ForegroundColor Cyan
git diff --cached --stat

Write-Host "`nCommitting..." -ForegroundColor Cyan
git commit -m $Message

if ($LASTEXITCODE -ne 0) {
    throw "git commit failed."
}

Write-Host "`nPushing to origin/main..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -ne 0) {
    throw "git push failed."
}

Write-Host "`nGitHub sync complete." -ForegroundColor Green
