"""파이프라인 orchestration.

Stage별 구현을 직접 가지지 않고 다음 모듈을 조립한다.
- preprocessing.py: load/crop/dataset iteration
- detection.py: Stage 1
- encoder.py: Stage 2
- prototype.py: Stage 3
- scoring.py: Stage 4~5
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_split_arrays(
    split_dir: Path | str,
    cap: Optional[int] = None,
) -> Tuple[List, List[str], List[str]]:
    """split 디렉터리 → 이미지, 라벨, 경로."""
    from preprocessing import iter_split, load_image

    images, labels, paths = [], [], []
    per_class: Dict[str, int] = {}

    for path, label in iter_split(Path(split_dir)):
        if cap is not None and per_class.get(label, 0) >= cap:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        images.append(load_image(path))
        labels.append(label)
        paths.append(str(path))

    return images, labels, paths


def embed_split(
    split_dir: Path | str,
    out: Path | str,
    cap: Optional[int] = None,
) -> Tuple[int, tuple]:
    """Stage 2 실행: crop 완료 split → embedding npz."""
    import numpy as np

    from encoder import BreedEncoder

    images, labels, paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    embeddings = np.asarray(BreedEncoder().encode_batch(images))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        labels=np.asarray(labels),
        paths=np.asarray(paths),
    )
    return len(images), embeddings.shape


def build_and_save_prototypes(
    embeddings_npz: Path | str,
    out: Path | str,
    cap: int = 50,
) -> List[str]:
    """Stage 3 실행: embedding npz → prototype npz."""
    import numpy as np

    from prototype import build_prototypes

    data = np.load(embeddings_npz, allow_pickle=True)
    embeddings, labels = data["embeddings"], data["labels"]

    embs_by_class: Dict[str, List] = {}
    for embedding, label in zip(embeddings, labels):
        embs_by_class.setdefault(str(label), []).append(embedding)

    result = build_prototypes(embs_by_class, cap=cap)
    prototypes = result["prototypes"]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    breeds = sorted(prototypes)
    np.savez_compressed(
        out,
        breeds=np.asarray(breeds),
        prototypes=np.stack([np.asarray(prototypes[breed]) for breed in breeds]),
        global_mean=np.asarray(result["global_mean"]),
    )
    return breeds


def load_prototypes(path: Path | str) -> dict:
    """prototype npz → scoring 모듈이 사용하는 dict."""
    import numpy as np

    data = np.load(path, allow_pickle=True)
    breeds = [str(breed) for breed in data["breeds"]]
    return {
        "prototypes": {
            breed: data["prototypes"][index]
            for index, breed in enumerate(breeds)
        },
        "global_mean": data["global_mean"],
    }


def embed_single_image(
    image_path: Path | str,
    *,
    detector=None,
    encoder=None,
):
    """Inference 앞단: detect → standard crop → encode.

    detector/encoder 주입을 허용해 반복 추론 시 모델을 매번 다시 로드하지 않게 한다.
    """
    from detection import DogDetector
    from encoder import BreedEncoder
    from preprocessing import crop_dog, load_image

    detector = detector or DogDetector()
    encoder = encoder or BreedEncoder()

    image = load_image(image_path)
    detections = detector.detect(image)
    if not detections:
        return None

    bbox, _confidence = detections[0]  # detect()가 confidence 내림차순 보장
    crop = crop_dog(image, bbox)
    return encoder.encode(crop)


def infer_image(
    image_path: Path | str,
    prototypes_path: Path | str,
    threshold: float,
    top_k: int = 3,
    temperature: float = 0.1,
    *,
    detector=None,
    encoder=None,
) -> Optional[dict]:
    """단일 이미지 전체 inference orchestration."""
    from scoring import predict

    embedding = embed_single_image(
        image_path,
        detector=detector,
        encoder=encoder,
    )
    if embedding is None:
        return None

    prototypes = load_prototypes(prototypes_path)
    return predict(
        embedding,
        prototypes,
        threshold=threshold,
        top_k=top_k,
        temperature=temperature,
    )
