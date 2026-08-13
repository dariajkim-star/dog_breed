"""Task 2+4 — Dataset Inventory + Master Manifest 생성.

각 소스(Stanford / Oxford / Tsinghua)를 스캔해 단일 manifest로 통합한다.
- bbox 내부 표준: xyxy absolute pixel
- 이상 데이터는 삭제하지 않고 usable=False + exclusion_reason 기록
- md5는 여기서 계산 (pHash/embedding dedup은 별도 스크립트)

사용:  python scripts/build_manifest.py
출력:  data/manifests/master_manifest.parquet, data/reports/dataset_inventory.md
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
REPORTS = ROOT / "data" / "reports"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size, None  # (w, h)
    except Exception as e:
        return (0, 0), f"corrupt_image:{type(e).__name__}"


def parse_voc_bbox(xml_path: Path, names=("bndbox",)):
    """VOC xml에서 bbox(xyxy) 목록 추출. names로 headbndbox/bodybndbox 구분."""
    out = {n: [] for n in names}
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    for obj in root.iter("object"):
        for n in names:
            for bb in obj.iter(n):
                box = [
                    float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                    float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")),
                ]
                out[n].append(box)
    return out


def validate_bbox(box, w, h):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return "invalid_bbox_order"
    if x2 > w * 1.02 or y2 > h * 1.02 or x1 < -w * 0.02 or y1 < -h * 0.02:
        return "bbox_out_of_image"
    return None


def scan_stanford(rows: list):
    img_root = RAW / "stanford" / "Images"
    ann_root = RAW / "stanford" / "Annotation"
    for breed_dir in tqdm(sorted(img_root.iterdir()), desc="stanford"):
        if not breed_dir.is_dir():
            continue
        breed = breed_dir.name.split("-", 1)[1]
        for img in breed_dir.glob("*.jpg"):
            (w, h), err = image_size(img)
            ann = ann_root / breed_dir.name / img.stem
            boxes = parse_voc_bbox(ann, ("bndbox",)) if ann.exists() else None
            body = boxes["bndbox"][0] if boxes and boxes["bndbox"] else None
            reason = err or (None if body else "missing_annotation")
            reason = reason or validate_bbox(body, w, h)
            rows.append(dict(
                image_id=f"stanford/{breed_dir.name}/{img.name}", source="stanford",
                image_path=str(img.relative_to(ROOT)), original_breed=breed,
                width=w, height=h, body_bbox=body, head_bbox=None,
                n_dogs=len(boxes["bndbox"]) if boxes else 0,
                usable=reason is None, exclusion_reason=reason,
            ))


def scan_oxford(rows: list):
    img_root = RAW / "oxford" / "images"
    xml_root = RAW / "oxford" / "annotations" / "xmls"
    for img in tqdm(sorted(img_root.glob("*.jpg")), desc="oxford"):
        base = img.stem
        breed = base.rsplit("_", 1)[0]
        is_cat = breed[0].isupper()  # Oxford 규칙: 고양이는 대문자 시작
        (w, h), err = image_size(img)
        boxes = parse_voc_bbox(xml_root / f"{base}.xml", ("bndbox",))
        head = boxes["bndbox"][0] if boxes and boxes["bndbox"] else None  # Oxford bbox = head ROI
        reason = err  # head 없음은 정상(trainval만 xml 존재) → 제외 사유 아님
        rows.append(dict(
            image_id=f"oxford/{img.name}", source="oxford",
            image_path=str(img.relative_to(ROOT)),
            original_breed=breed.lower() if not is_cat else f"CAT_{breed}",
            width=w, height=h, body_bbox=None, head_bbox=head,
            n_dogs=0 if is_cat else 1,
            usable=(reason is None) and not is_cat,  # 고양이는 OOD 전용 → usable=False, ood set에서 별도 사용
            exclusion_reason=reason or ("cat_ood_only" if is_cat else None),
        ))


def scan_tsinghua(rows: list):
    # 이미지: low-resolution/<breed>/..., annotation: annotations/Low-Annotations/<breed>/<img>.xml
    img_root = next(p for p in (RAW / "tsinghua").iterdir()
                    if p.is_dir() and p.name.lower().startswith("low") and "annot" not in p.name.lower())
    ann_root = RAW / "tsinghua" / "annotations" / "Low-Annotations"
    for breed_dir in tqdm(sorted(img_root.iterdir()), desc="tsinghua"):
        if not breed_dir.is_dir():
            continue
        breed = breed_dir.name.split("-", 2)[-1]
        ann_dir = ann_root / breed_dir.name
        imgs = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.jpeg"))
        for img in imgs:
            (w, h), err = image_size(img)
            boxes = parse_voc_bbox(ann_dir / f"{img.name}.xml", ("bodybndbox", "headbndbox"))
            body = boxes["bodybndbox"][0] if boxes and boxes["bodybndbox"] else None
            head = boxes["headbndbox"][0] if boxes and boxes["headbndbox"] else None
            reason = err or (None if body else "missing_annotation")
            reason = reason or validate_bbox(body, w, h) or validate_bbox(head, w, h)
            rows.append(dict(
                image_id=f"tsinghua/{breed_dir.name}/{img.name}", source="tsinghua",
                image_path=str(img.relative_to(ROOT)), original_breed=breed,
                width=w, height=h, body_bbox=body, head_bbox=head,
                n_dogs=len(boxes["bodybndbox"]) if boxes else 0,
                usable=reason is None, exclusion_reason=reason,
            ))


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list = []
    scan_stanford(rows)
    scan_oxford(rows)
    scan_tsinghua(rows)
    df = pd.DataFrame(rows)

    print("computing md5 ...")
    df["md5"] = [md5_of(ROOT / p) for p in tqdm(df["image_path"])]

    df.to_parquet(MANIFESTS / "master_manifest.parquet", index=False)

    # ---- inventory report ----
    inv = df.groupby("source").agg(
        images=("image_id", "count"),
        breeds=("original_breed", "nunique"),
        with_body_bbox=("body_bbox", lambda s: s.notna().sum()),
        with_head_bbox=("head_bbox", lambda s: s.notna().sum()),
        usable=("usable", "sum"),
    )
    lines = ["# Dataset Inventory (auto-generated)", "", inv.to_markdown(), "",
             "## Exclusion reasons", "",
             df[~df.usable].groupby(["source", "exclusion_reason"]).size().to_markdown(), "",
             "## Breed x source image counts (top 40)", "",
             df[df.usable].groupby(["source", "original_breed"]).size()
               .sort_values(ascending=False).head(40).to_markdown()]
    (REPORTS / "dataset_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(inv)
    print(f"\nmanifest rows: {len(df)}  -> {MANIFESTS / 'master_manifest.parquet'}")


if __name__ == "__main__":
    main()
