# =============================================================
# Dog Breed Project — Dataset Download Script (Task 1)
# 사용: powershell -File scripts\download_all.ps1 [-Target tsinghua|stanford|oxford|all]
# curl.exe -C - 로 중단 시 이어받기 지원. 재실행해도 안전.
# =============================================================
param([string]$Target = "all")

$root = Split-Path $PSScriptRoot -Parent
$raw  = Join-Path $root "data\raw"

function Get-File($url, $outDir, $name) {
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $out = Join-Path $outDir $name
    Write-Host ">>> $name  <=  $url"
    & curl.exe -L -C - --retry 5 --retry-delay 10 -o $out $url
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! FAILED: $name (exit $LASTEXITCODE)" }
}

# ---------- Tsinghua Dogs (저해상도판 + annotation + split) ----------
if ($Target -in "all","tsinghua") {
    $d = Join-Path $raw "tsinghua"
    Get-File "https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip"      $d "TrainValSplit.zip"
    Get-File "https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip"    $d "low-annotations.zip"
    Get-File "https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1"   $d "low-resolution.zip"   # 2.5GB
}

# ---------- Stanford Dogs (images 757MB + VOC annotation + split .mat) ----------
if ($Target -in "all","stanford") {
    $d = Join-Path $raw "stanford"
    Get-File "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"      $d "lists.tar"
    Get-File "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar" $d "annotation.tar"
    Get-File "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"     $d "images.tar"
}

# ---------- Oxford-IIIT Pet (images ~790MB + trimap/head bbox annotation) ----------
if ($Target -in "all","oxford") {
    $d = Join-Path $raw "oxford"
    Get-File "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" $d "annotations.tar.gz"
    Get-File "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"      $d "images.tar.gz"
}

Write-Host "=== download_all.ps1 done (Target=$Target) ==="
# Open Images V7 / COCO 는 FiftyOne 사용: scripts\download_oiv7_coco.py 참고
