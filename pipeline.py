"""파이프라인 단계 실행 로직 — CLI(main.py)와 분리된 순수 함수 모음.

각 함수는 파이프라인의 한 단계만 수행한다 (README 아키텍처 참조).
    load_split_arrays  : split 폴더 → (이미지, 라벨, 경로)
    embed_split        : split → embedding npz 저장 (Stage 2)
    build_and_save_prototypes : embedding npz → prototype npz (Stage 3)
    load_prototypes    : prototype npz → model 함수용 dict 복원
    embed_single_image : detect → crop → encode (README 4 추론 앞단)
    infer_image        : 단일 이미지 전체 경로 실행 (README 4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_split_arrays(
    split_dir: Path | str, cap: Optional[int] = None
) -> Tuple[List, List[str], List[str]]:
    """split 디렉토리를 순회해 (이미지 리스트, 라벨, 경로 문자열)을 모은다.

    processed 데이터는 이미 518px 정사각 crop이므로 detection 불필요
    (docs/handover.md 기능 2 참조). cap 지정 시 클래스당 앞에서부터 cap장만
    사용 (CPU 테스트 런 용도 — prototype은 어차피 클래스당 ≤50장).
    """
    from preprocessing import load_image, iter_split

    imgs, labels, paths = [], [], []
    per_class: Dict[str, int] = {}
    for path, label in iter_split(Path(split_dir)):
        if cap is not None and per_class.get(label, 0) >= cap:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        imgs.append(load_image(path))
        labels.append(label)
        paths.append(str(path))
    return imgs, labels, paths


def embed_split(
    split_dir: Path | str, out: Path | str, cap: Optional[int] = None
) -> Tuple[int, tuple]:
    """split 이미지를 embedding으로 변환해 npz 저장. (장수, embedding shape) 반환."""
    import numpy as np
    from model import BreedEncoder

    imgs, labels, paths = load_split_arrays(split_dir, cap=cap)
    if not imgs:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    embeddings = np.asarray(BreedEncoder().encode_batch(imgs))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        labels=np.asarray(labels),
        paths=np.asarray(paths),
    )
    return len(imgs), embeddings.shape


def build_and_save_prototypes(
    embeddings_npz: Path | str, out: Path | str, cap: int = 50
) -> List[str]:
    """embedding npz → 견종별 prototype 생성 후 npz 저장. 견종 목록 반환."""
    import numpy as np
    from model import build_prototypes

    data = np.load(embeddings_npz, allow_pickle=True)
    embeddings, labels = data["embeddings"], data["labels"]

    # 견종별로 embedding을 모은다 (클래스당 cap장 캡 — README 3 참조)
    embs_by_class: Dict[str, List] = {}
    for emb, label in zip(embeddings, labels):
        embs_by_class.setdefault(str(label), []).append(emb)

    result = build_prototypes(embs_by_class, cap=cap)
    protos = result["prototypes"]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    breeds = sorted(protos.keys())
    np.savez_compressed(
        out,
        breeds=np.asarray(breeds),
        prototypes=np.stack([np.asarray(protos[b]) for b in breeds]),
        global_mean=np.asarray(result["global_mean"]),
    )
    return breeds


def load_prototypes(path: Path | str) -> dict:
    """prototype npz를 model 함수들이 기대하는 dict 형태로 복원한다."""
    import numpy as np

    data = np.load(path, allow_pickle=True)
    breeds = [str(b) for b in data["breeds"]]
    return {
        "prototypes": {b: data["prototypes"][i] for i, b in enumerate(breeds)},
        "global_mean": data["global_mean"],
    }


def embed_single_image(image_path: Path | str):
    """단일 이미지: detect → crop_dog → encode. embedding(384,) 반환, 실패 시 None.

    ⚠️ crop 규칙은 prototype 구축과 완전 동일해야 함 (README 1. 제1원칙).
    """
    from preprocessing import load_image, crop_dog
    from model import DogDetector, BreedEncoder

    img = load_image(image_path)
    detections = DogDetector().detect(img)
    if not detections:
        return None
    # 가장 confidence 높은 bbox 사용
    bbox, _conf = max(detections, key=lambda d: d[1])
    crop = crop_dog(img, bbox)
    return BreedEncoder().encode(crop)


def infer_image(
    image_path: Path | str,
    prototypes_path: Path | str,
    threshold: float,
    top_k: int = 3,
    temperature: float = 0.1,
) -> Optional[dict]:
    """단일 이미지 전체 경로 실행 (README 4). detection 실패 시 None."""
    from model import predict

    emb = embed_single_image(image_path)
    if emb is None:
        return None
    proto = load_prototypes(prototypes_path)
    return predict(emb, proto, threshold=threshold, top_k=top_k, temperature=temperature)
