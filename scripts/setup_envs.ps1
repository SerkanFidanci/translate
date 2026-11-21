# dubpoc environment setup (Windows / PowerShell 5.1+)
# (ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1 -Only core,tts
#   optional XTTS-v2 env (CPML non-commercial license, see README): -Only xtts
#
# Every heavy model gets its own venv (spec 1.1). torch wheels come from the right
# CUDA index via uv's --torch-backend; envs that share a torch version are
# hard-linked from the uv cache, so they do not cost disk space twice.
param(
    [string[]]$Only = @("core", "dnr", "sep", "diar", "emo", "tts", "vc", "omni"),
    [string]$Torch = "cu126",
    [string]$SeedVcCommit = "51383efd921027683c89e5348211d93ff12ac2a8",
    [string]$MsstCommit = "1f700c386e0dd763fe0a57242bb9420dce987354"
)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# -File passes "core,tts" as a single string; split it.
$Only = @($Only | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install: winget install astral-sh.uv  (https://docs.astral.sh/uv/)"
}

$failed = @()
foreach ($name in $Only) {
    Write-Host "==> [$name] venv" -ForegroundColor Cyan
    $py = Join-Path $Root "envs\$name\Scripts\python.exe"
    # (uv venv --allow-existing fails on Windows when python.exe already exists)
    if (-not (Test-Path $py)) {
        uv venv "envs\$name" --python 3.11 --clear
        if ($LASTEXITCODE -ne 0) { $failed += $name; continue }
    }
    switch ($name) {
        "core" {
            uv pip install --python $py -e ".[asr,dev]"
        }
        "dnr" {
            if (-not (Test-Path "tools\msst\.git")) {
                git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git tools\msst
            }
            git -C tools\msst fetch -q --depth 1 origin $MsstCommit
            git -C tools\msst checkout -q $MsstCommit
            uv pip install --python $py --torch-backend $Torch -r requirements\dnr.txt
            # Bandit v2 multilingual DnR weights (Zenodo 12701995, CC BY-SA 4.0, ~447 MB)
            $ck = "tools\models\bandit_v2\checkpoint-multi.ckpt"
            if (-not (Test-Path $ck)) {
                New-Item -ItemType Directory -Force "tools\models\bandit_v2" | Out-Null
                curl.exe -L --retry 3 -o $ck "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1"
            }
            $md5 = (Get-FileHash $ck -Algorithm MD5).Hash.ToLower()
            if ($md5 -ne "fea2868787551b0cff36cfcf7c3622a3") { Write-Host "checkpoint MD5 mismatch: $md5" -ForegroundColor Red; $failed += $name; continue }
        }
        "vc" {
            if (-not (Test-Path "tools\seed-vc\.git")) {
                git clone https://github.com/Plachtaa/seed-vc.git tools\seed-vc
            }
            git -C tools\seed-vc fetch -q --depth 1 origin $SeedVcCommit
            git -C tools\seed-vc checkout -q $SeedVcCommit
            uv pip install --python $py --torch-backend $Torch -r requirements\vc.txt
        }
        default {
            uv pip install --python $py --torch-backend $Torch -r "requirements\$name.txt"
        }
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> [$name] FAILED" -ForegroundColor Red
        $failed += $name
    } else {
        Write-Host "==> [$name] ok" -ForegroundColor Green
    }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Note: no system ffmpeg on PATH. dubpoc falls back to the ffmpeg bundled with imageio-ffmpeg." -ForegroundColor Yellow
    Write-Host "      For a full build (incl. rubberband filter): winget install Gyan.FFmpeg" -ForegroundColor Yellow
}
if ($failed.Count -gt 0) {
    Write-Host ("Failed envs: " + ($failed -join ", ")) -ForegroundColor Red
    exit 1
}
Write-Host "Done. Check with:  envs\core\Scripts\dubpoc.exe doctor"
