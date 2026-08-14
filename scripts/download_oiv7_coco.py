"""Open Images V7 (Dog) + COCO 2017 (dog) 다운로드 — FiftyOne 사용.

사전 설치: pip install fiftyone
사용:
    python scripts/download_oiv7_coco.py oiv7   # Open Images V7 dog (detection 학습용)
    python scripts/download_oiv7_coco.py coco   # COCO val2017 dog (detection 외부 평가용)

OIv7는 dog 클래스만 받아도 수만 장이므로 max_samples로 1차 제한.
IsGroupOf / IsDepiction 필터링은 이후 전처리 단계(detection dataset 생성)에서 수행.
"""
import sys

import fiftyone as fo
import fiftyone.zoo as foz

EXPORT_ROOT = "data/raw"


def download_oiv7(max_samples: int = 20000) -> None:
    for split in ("train", "validation"):
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=["Dog"],
            max_samples=max_samples if split == "train" else 3000,
            dataset_name=f"oiv7-dog-{split}",
        )
        label_field = next(
            f for f in ("detections", "ground_truth")
            if f in ds.get_field_schema()
        )
        print(split, len(ds), "samples, label_field =", label_field)
        ds.export(
            export_dir=f"{EXPORT_ROOT}/open_images/{split}",
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
            classes=["Dog"],
        )


def download_coco() -> None:
    ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["dog"],
        dataset_name="coco-dog-val",
    )
    print("coco val", len(ds), "samples")
    ds.export(
        export_dir=f"{EXPORT_ROOT}/coco/validation",
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        classes=["dog"],
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "oiv7"
    if target == "oiv7":
        download_oiv7()
    elif target == "coco":
        download_coco()
    else:
        raise SystemExit(f"unknown target: {target}")
