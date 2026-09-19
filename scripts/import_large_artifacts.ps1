param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,

    [switch]$CommitAndPush
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ManifestPath = Join-Path $RepoRoot "artifacts_manifest.csv"

if (-not (Test-Path $ArchivePath)) {
    throw "Archive not found: $ArchivePath"
}

if (-not (Test-Path $ManifestPath)) {
    throw "Missing artifacts manifest: $ManifestPath"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not available in PATH."
}

try {
    git lfs version | Out-Null
}
catch {
    throw "Git LFS is required. Install Git LFS, then rerun this script."
}

Write-Step "Preparing temporary extraction directory"
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rl_robot_import_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot | Out-Null

try {
    Write-Step "Extracting outer archive"
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $TempRoot -Force

    $NestedRobot = Get-ChildItem -Path $TempRoot -Recurse -Filter "automous-robot.zip" | Select-Object -First 1
    $NestedRestaurant = Get-ChildItem -Path $TempRoot -Recurse -Filter "restaurant.zip" | Select-Object -First 1

    if (-not $NestedRobot) {
        throw "Could not find automous-robot.zip inside the supplied archive."
    }
    if (-not $NestedRestaurant) {
        throw "Could not find restaurant.zip inside the supplied archive."
    }

    $RobotExtract = Join-Path $TempRoot "robot_project"
    $RestaurantExtract = Join-Path $TempRoot "restaurant_project"

    Write-Step "Extracting robot project"
    Expand-Archive -LiteralPath $NestedRobot.FullName -DestinationPath $RobotExtract -Force

    Write-Step "Extracting restaurant project"
    Expand-Archive -LiteralPath $NestedRestaurant.FullName -DestinationPath $RestaurantExtract -Force

    $RobotSource = Join-Path $RobotExtract "robot"
    $RestaurantSource = Join-Path $RestaurantExtract "restaurant"

    if (-not (Test-Path $RobotSource)) {
        throw "Expected robot folder was not found after extraction."
    }
    if (-not (Test-Path $RestaurantSource)) {
        throw "Expected restaurant folder was not found after extraction."
    }

    Write-Step "Copying original project contents into repository"

    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "robot") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "restaurant") | Out-Null

    Copy-Item -Path (Join-Path $RobotSource "*") -Destination (Join-Path $RepoRoot "robot") -Recurse -Force
    Copy-Item -Path (Join-Path $RestaurantSource "*") -Destination (Join-Path $RepoRoot "restaurant") -Recurse -Force

    Get-ChildItem -Path (Join-Path $RepoRoot "robot") -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Get-ChildItem -Path (Join-Path $RepoRoot "restaurant") -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force

    Write-Step "Initializing Git LFS"
    Push-Location $RepoRoot
    try {
        git lfs install
        git lfs track "*.zip"
        git lfs track "*.pkl"
        git lfs track "*.npz"
        git lfs track "*.tfevents.*"
    }
    finally {
        Pop-Location
    }

    Write-Step "Verifying binary artifacts against SHA-256 manifest"
    $Manifest = Import-Csv $ManifestPath
    $Failures = @()

    foreach ($Row in $Manifest) {
        $RelativePath = $Row.path -replace "/", [System.IO.Path]::DirectorySeparatorChar
        $LocalPath = Join-Path $RepoRoot $RelativePath

        if (-not (Test-Path $LocalPath)) {
            $Failures += "MISSING: $($Row.path)"
            continue
        }

        $Size = (Get-Item $LocalPath).Length
        if ([int64]$Size -ne [int64]$Row.size_bytes) {
            $Failures += "SIZE MISMATCH: $($Row.path)"
            continue
        }

        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $LocalPath).Hash.ToLowerInvariant()
        if ($Hash -ne $Row.sha256.ToLowerInvariant()) {
            $Failures += "HASH MISMATCH: $($Row.path)"
        }
    }

    if ($Failures.Count -gt 0) {
        Write-Host ""
        Write-Host "Artifact verification failed:" -ForegroundColor Red
        $Failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        throw "One or more binary artifacts did not match the uploaded project."
    }

    Write-Host "All $($Manifest.Count) binary artifacts verified." -ForegroundColor Green

    Write-Step "Staging project files"
    Push-Location $RepoRoot
    try {
        git add .gitattributes
        git add robot restaurant

        Write-Host ""
        git status --short
        Write-Host ""

        if ($CommitAndPush) {
            Write-Step "Committing full training artifacts"
            git commit -m "Add complete RL models, replay buffers and training artifacts"

            Write-Step "Pushing to GitHub"
            git push origin main
        }
        else {
            Write-Host "Files are verified and staged." -ForegroundColor Green
            Write-Host "Review them with: git status"
            Write-Host "Then commit and push when ready:"
            Write-Host '  git commit -m "Add complete RL models, replay buffers and training artifacts"'
            Write-Host "  git push origin main"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if (Test-Path $TempRoot) {
        Remove-Item -Recurse -Force $TempRoot
    }
}
