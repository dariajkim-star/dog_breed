"""processed/ export — 기능별 최종 학습 데이터 생성 (split FREEZE 반영).

processed/
├── detection/  images/train|val|test/ + labels/... + data.yaml   (기능1: 검출)
│     train/val = OIv7, test = COCO. hardlink. YOLO 공식 포맷
│     (https://docs.ultralytics.com/datasets/detect/)
├── breed_body/ train|val|test/<breed>/*.jpg   (기능2: 견종 - body crop 518px)
├── breed_head/ train|val|test/<breed>/*.jpg   (tsinghua head crop)
└── ood/        val|test/cat/*.jpg             (Oxford 고양이, 원본 hardlink)

사용: python scripts/export_processed.py [detection|breed|ood|all]
"""
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.crop import standard_crop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def export_detection():
    plan = [("open_images/train", "train"), ("open_images/validation", "val"),
            ("coco/validation", "test")]
    for src_sub, split in plan:
        src = ROOT / "data" / "labeled" / src_sub
        n = 0
        for img in tqdm(list((src / "images").rglob("*.*")), desc=f"det/{split}"):
            lbl = src / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            hardlink(img, PROC / "detection" / "images" / split / img.name)
            hardlink(lbl, PROC / "detection" / "labels" / split / lbl.name)
            n += 1
        print(f"detection/{split}: {n}")
    (PROC / "detection" / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: dog\n", encoding="utf-8")


def export_breed():
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    mvp = df[df.in_mvp25 & df.split.notna()]
    n_body = n_head = n_fail = 0
    for row in tqdm(mvp.itertuples(), total=len(mvp), desc="breed crops"):
        try:
            with Image.open(ROOT / row.image_path) as im:
                im.load()
                fname = Path(row.image_path).stem + ".jpg"
                if row.body_bbox is not None:
                    crop = standard_crop(im, row.body_bbox)
                    out = PROC / "breed_body" / row.split / row.canonical_breed / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out, quality=92)
                    n_body += 1
                if row.head_bbox is not None:
                    crop = standard_crop(im, row.head_bbox)
                    out = PROC / "breed_head" / row.split / row.canonical_breed / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out, quality=92)
                    n_head += 1
        except Exception:
            n_fail += 1
    print(f"breed_body: {n_body}, breed_head: {n_head}, failed: {n_fail}")


def export_ood():
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    cats = df[(df.canonical_breed == "__CAT_OOD__") & df.split.notna()]
    for row in tqdm(cats.itertuples(), total=len(cats), desc="ood cats"):
        src = ROOT / row.image_path
        if src.exists():
            hardlink(src, PROC / "ood" / row.split / "cat" / Path(row.image_path).name)
    print(f"ood cats: {len(cats)}")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "all"
    if t in ("detection", "all"):
        export_detection()
    if t in ("ood", "all"):
        export_ood()
    if t in ("breed", "all"):
        export_breed()
