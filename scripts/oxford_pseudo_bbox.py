"""Oxford 개 이미지에 YOLO11n(COCO pretrained)으로 body pseudo-bbox 생성.

- 대상: manifest에서 source=oxford, usable=True (개만 — 고양이 제외)
- COCO class 16 = dog
- 다중 검출 시: 최고 confidence 1개 채택 (Oxford는 이미지당 개 1마리 전제)
- 결과: data/manifests/oxford_pseudo_bbox.parquet
  (image_id, pseudo_body_bbox[xyxy abs], conf, n_det, needs_review)
- needs_review 기준: conf < 0.5 또는 검출 0건 → 육안 스팟체크 대상

사용:  python scripts/oxford_pseudo_bbox.py
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DOG_CLASS = 16  # COCO
CONF_REVIEW = 0.5


def main():
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    ox = df[(df.source == "oxford") & df.usable]
    print(f"oxford dog images: {len(ox)}")

    # 여기는 일부러 11n으로 고정한다. oxford_pseudo_bbox.parquet에 이미 들어간
    # bbox가 11n으로 만들어진 값이라, 모델을 바꾸면 재실행 시 기존 manifest와
    # 다른 결과가 나와 데이터 재현성이 깨진다. (추론 기본값은 11s로 올렸다 — detection.py 참조)
    model = YOLO("yolo11n.pt")
    rows = []
    paths = [str(ROOT / p) for p in ox.image_path]
    ids = ox.image_id.tolist()

    BATCH = 32
    for i in tqdm(range(0, len(paths), BATCH)):
        results = model.predict(paths[i:i + BATCH], classes=[DOG_CLASS],
                                conf=0.1, verbose=False)
        for image_id, r in zip(ids[i:i + BATCH], results):
            if len(r.boxes) == 0:
                rows.append(dict(image_id=image_id, pseudo_body_bbox=None,
                                 conf=0.0, n_det=0, needs_review=True))
                continue
            k = int(r.boxes.conf.argmax())
            conf = float(r.boxes.conf[k])
            box = [round(float(v), 1) for v in r.boxes.xyxy[k]]
            rows.append(dict(image_id=image_id, pseudo_body_bbox=box, conf=conf,
                             n_det=len(r.boxes),
                             needs_review=conf < CONF_REVIEW))

    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data" / "manifests" / "oxford_pseudo_bbox.parquet", index=False)
    print("\n=== summary ===")
    print("total:", len(out))
    print("detected:", int((out.n_det > 0).sum()))
    print("no detection:", int((out.n_det == 0).sum()))
    print("needs_review (conf<0.5 or none):", int(out.needs_review.sum()))
    print("multi-dog images (n_det>1):", int((out.n_det > 1).sum()))
    print("mean conf:", round(float(out.conf.mean()), 3))


if __name__ == "__main__":
    main()
