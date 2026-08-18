"""preprocessing.py — 이미지 로딩·crop 위임·데이터셋 순회 유틸.

crop 로직은 utils/crop.py의 standard_crop()이 유일한 구현(README '제1원칙')이며,
이 모듈은 numpy <-> PIL 변환과 파일 순회만 담당한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

from utils.crop import standard_crop

# 지원하는 이미지 확장자 (소문자 비교)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """이미지를 RGB HWC uint8 배열로 로드한다.

    EXIF Orientation 태그를 반영해 픽셀을 실제로 회전시킨다 —
    이를 빼면 스마트폰 세로 사진의 bbox 좌표가 어긋난다.
    공식 스펙: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose
    """
    with Image.open(path) as im:
        # exif_transpose: Orientation 태그대로 회전/반전 후 태그 제거
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def crop_dog(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """dog bbox(x1, y1, x2, y2, 절대 픽셀)를 표준 규칙으로 crop한다.

    규칙 전체(15% expand → square → clamp → gray-114 pad → 518px resize)는
    standard_crop()의 기본값에 위임 — 여기서 파라미터를 덮어쓰지 않는다.
    """
    pil = Image.fromarray(img)  # HWC uint8 → PIL RGB
    cropped = standard_crop(pil, bbox)  # 프로젝트 기본값 그대로 사용
    return np.asarray(cropped, dtype=np.uint8)


def iter_split(split_dir: str | Path) -> Iterator[tuple[Path, str]]:
    """클래스별 하위 폴더 구조의 split 디렉터리를 순회한다.

    구조 예: data/processed/breed_body/train/<class>/*.jpg
    (image_path, class_label)을 클래스명·파일명 정렬 순서로 yield한다.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split 디렉터리가 없습니다: {split_dir}")

    # 클래스 폴더만 골라 이름순으로 정렬 — 실행할 때마다 같은 순서 보장
    class_dirs = []
    for entry in split_dir.iterdir():
        if entry.is_dir():
            class_dirs.append(entry)
    class_dirs.sort()

    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTS:
                yield img_path, label
