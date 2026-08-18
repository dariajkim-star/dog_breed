"""download_all.ps1로 받은 압축 파일을 manifest가 기대하는 경로로 푼다.

download_all.ps1은 다운로드만 하고 압축을 풀지 않는다. build_manifest.py와
export_processed.py는 master_manifest.parquet의 image_path를 그대로 읽으므로
아래 구조가 정확히 맞아야 한다:

    data/raw/stanford/Images/<breed>/*.jpg
    data/raw/stanford/Annotation/<breed>/*
    data/raw/oxford/images/*.jpg
    data/raw/oxford/annotations/xmls/*.xml
    data/raw/tsinghua/low-resolution/<breed>/*.jpg
    data/raw/tsinghua/annotations/Low-Annotations/<breed>/*.xml

압축 파일 안의 최상위 폴더 이름이 기대와 다를 수 있으므로, 풀어놓고 나서
실제 경로를 찾아 필요하면 이름을 바꾼다.

사용:  python scripts/extract_raw.py
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

# Windows cp949 콘솔에서 한글·기호가 깨지지 않도록 (commands.py와 동일한 처리)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# (압축파일, 풀 위치, 다 풀고 나면 있어야 하는 경로)
JOBS = [
    ("stanford/images.tar",            "stanford", "stanford/Images"),
    ("stanford/annotation.tar",        "stanford", "stanford/Annotation"),
    ("stanford/lists.tar",             "stanford", None),
    ("oxford/images.tar.gz",           "oxford",   "oxford/images"),
    ("oxford/annotations.tar.gz",      "oxford",   "oxford/annotations"),
    ("tsinghua/low-resolution.zip",    "tsinghua", "tsinghua/low-resolution"),
    ("tsinghua/low-annotations.zip",   "tsinghua", "tsinghua/annotations"),
    ("tsinghua/TrainValSplit.zip",     "tsinghua", None),
]


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:  # .tar / .tar.gz
        with tarfile.open(archive) as t:
            # data 필터: 절대경로/상위경로 탈출 항목을 막는 파이썬 기본 안전장치
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # python < 3.12
                t.extractall(dest)


def main() -> None:
    if not RAW.is_dir():
        sys.exit(f"[!] {RAW} 가 없습니다. download_all.ps1 을 먼저 실행하세요.")

    failed: list[tuple[str, str]] = []
    for rel_archive, rel_dest, rel_expect in JOBS:
        archive = RAW / rel_archive
        dest = RAW / rel_dest
        if not archive.exists():
            print(f"[건너뜀] 압축 파일 없음: {rel_archive}")
            continue
        if rel_expect and (RAW / rel_expect).is_dir():
            print(f"[건너뜀] 이미 풀려 있음: {rel_expect}")
            continue
        size_mb = archive.stat().st_size / 1e6
        print(f"[풀기] {rel_archive} ({size_mb:,.0f}MB) -> data/raw/{rel_dest}/")
        try:
            extract(archive, dest)
        except Exception as e:
            # 압축 파일 하나가 깨져도 나머지는 계속 푼다 (다운로드 중 잘린 경우 등).
            # 잘린 파일은 curl -C - 로 이어받은 뒤 이 스크립트를 다시 실행하면 된다.
            print(f"   [!] 실패: {type(e).__name__}: {e}")
            failed.append((rel_archive, f"{type(e).__name__}: {e}"))

    # ---- Tsinghua 주석 폴더 위치 보정 ----
    # low-annotations.zip 안의 최상위 폴더 이름이 배포판마다 다르다.
    # build_manifest.py는 data/raw/tsinghua/annotations/Low-Annotations 를 본다.
    ts = RAW / "tsinghua"
    want = ts / "annotations" / "Low-Annotations"
    if not want.is_dir():
        found = None
        for cand in ts.rglob("Low-Annotations"):
            if cand.is_dir():
                found = cand
                break
        if found:
            want.parent.mkdir(parents=True, exist_ok=True)
            print(f"[이동] {found.relative_to(RAW)} -> {want.relative_to(RAW)}")
            shutil.move(str(found), str(want))

    # ---- 검증 ----
    print("\n=== 경로 검증 ===")
    # (경로, glob 패턴들, manifest 기준 기대 개수)
    # Tsinghua는 .jpg와 .jpeg가 섞여 있다 (build_manifest.py도 둘 다 훑는다).
    checks = [
        ("stanford/Images", ["*/*.jpg"], 20_580),
        ("stanford/Annotation", ["*/*"], 20_580),
        ("oxford/images", ["*.jpg"], 7_390),
        ("oxford/annotations/xmls", ["*.xml"], None),
        ("tsinghua/low-resolution", ["*/*.jpg", "*/*.jpeg"], 70_432),
        ("tsinghua/annotations/Low-Annotations", ["*/*.xml"], None),
    ]
    ok = True
    for rel, patterns, expect in checks:
        d = RAW / rel
        if not d.is_dir():
            print(f"  [없음] {rel}")
            ok = False
            continue
        n = sum(1 for p in patterns for _ in d.glob(p))
        if n == 0:
            print(f"  [비어있음] {rel}")
            ok = False
        elif expect is not None and n != expect:
            print(f"  [수량다름] {rel:40s} {n:>7,}개 (기대 {expect:,})")
            ok = False
        else:
            print(f"  [OK]   {rel:44s} {n:>7,}개")

    print("\n" + ("전부 정상 — export_processed.py 실행 가능"
                  if ok else "[!] 빠진 경로가 있습니다. 위 목록을 확인하세요."))


if __name__ == "__main__":
    main()
