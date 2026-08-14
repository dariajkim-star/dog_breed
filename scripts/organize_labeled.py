"""Raw data → 표준 라벨링 구조 정리.

data/labeled/<source>/images/...  (hardlink — 디스크 추가 사용 없음)
data/labeled/<source>/labels/...  (YOLO txt: class cx cy w h, normalized)

클래스 규약 (classes.txt):
  0 = dog_body
  1 = dog_head   (Tsinghua만 보유)

- tsinghua/stanford: manifest의 body/head bbox 사용, 견종 폴더 구조 유지
  → 품종 라벨 = 폴더명 / 위치 라벨 = labels/*.txt
- oxford: YOLO pseudo-bbox (oxford_pseudo_bbox.parquet) 사용, needs_review 제외
- open_images/coco: export된 labels.json 사용,
  IsGroupOf=True / IsDepiction=True annotation 제외, 필터 후 bbox 0개면 이미지도 제외

사용:
    python scripts/organize_labeled.py tsinghua stanford openimages coco
    python scripts/organize_labeled.py oxford      # pseudo-bbox 완료 후
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "labeled"


def hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def yolo_line(cls: int, box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_label(dst: Path, lines):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")


def do_manifest_source(source: str):
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    df = df[(df.source == source) & df.usable]
    n = 0
    for row in tqdm(df.itertuples(), total=len(df), desc=source):
        rel = Path(row.image_path)                 # data/raw/<source>/.../<breed_dir>/<file>
        breed_dir, fname = rel.parts[-2], rel.parts[-1]
        lines = []
        if row.body_bbox is not None:
            lines.append(yolo_line(0, row.body_bbox, row.width, row.height))
        if getattr(row, "head_bbox", None) is not None and row.head_bbox is not None:
            lines.append(yolo_line(1, row.head_bbox, row.width, row.height))
        if not lines:
            continue
        hardlink(ROOT / rel, OUT / source / "images" / breed_dir / fname)
        write_label(OUT / source / "labels" / breed_dir / (Path(fname).stem + ".txt"), lines)
        n += 1
    print(f"{source}: {n} images organized")


def do_oxford():
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    ox = df[(df.source == "oxford") & df.usable].set_index("image_id")
    pb = pd.read_parquet(ROOT / "data" / "manifests" / "oxford_pseudo_bbox.parquet")
    n = skip = 0
    for row in tqdm(pb.itertuples(), total=len(pb), desc="oxford"):
        if row.needs_review or row.pseudo_body_bbox is None:
            skip += 1
            continue
        m = ox.loc[row.image_id]
        rel = Path(m.image_path)
        fname = rel.parts[-1]
        breed = fname.rsplit("_", 1)[0]            # oxford는 평면 폴더 → 파일명에서 견종
        lines = [yolo_line(0, list(row.pseudo_body_bbox), m.width, m.height)]
        if m.head_bbox is not None:
            lines.append(yolo_line(1, m.head_bbox, m.width, m.height))
        hardlink(ROOT / rel, OUT / "oxford" / "images" / breed / fname)
        write_label(OUT / "oxford" / "labels" / breed / (Path(fname).stem + ".txt"), lines)
        n += 1
    print(f"oxford: {n} organized, {skip} skipped (needs_review) -> 육안 검토 목록은 parquet 참조")


def do_coco_style(source: str, subdirs):
    for sub in subdirs:
        base = ROOT / "data" / "raw" / source / sub
        with open(base / "labels.json", encoding="utf-8") as f:
            j = json.load(f)
        imgs = {im["id"]: im for im in j["images"]}
        by_img: dict = {}
        removed = 0
        for a in j["annotations"]:
            if a.get("IsGroupOf") or a.get("IsDepiction"):
                removed += 1
                continue
            by_img.setdefault(a["image_id"], []).append(a)
        n = 0
        for iid, anns in tqdm(by_img.items(), desc=f"{source}/{sub}"):
            im = imgs[iid]
            src = base / "data" / im["file_name"]
            if not src.exists():
                continue
            w, h = im["width"], im["height"]
            lines = [yolo_line(0, [a["bbox"][0], a["bbox"][1],
                                   a["bbox"][0] + a["bbox"][2],
                                   a["bbox"][1] + a["bbox"][3]], w, h) for a in anns]
            hardlink(src, OUT / source / sub / "images" / im["file_name"])
            write_label(OUT / source / sub / "labels" / (Path(im["file_name"]).stem + ".txt"), lines)
            n += 1
        print(f"{source}/{sub}: {n} images, {removed} annotations filtered (IsGroupOf/IsDepiction), "
              f"{len(imgs) - len(by_img)} images dropped (no valid box)")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["tsinghua", "stanford", "openimages", "coco"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.txt").write_text("0 dog_body\n1 dog_head\n", encoding="utf-8")
    for t in targets:
        if t in ("tsinghua", "stanford"):
            do_manifest_source(t)
        elif t == "oxford":
            do_oxford()
        elif t == "openimages":
            do_coco_style("open_images", ["train", "validation"])
        elif t == "coco":
            do_coco_style("coco", ["validation"])
