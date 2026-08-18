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
        # 이 클래스에서 지금까지 몇 장 담았는지 확인
        count_so_far = per_class.get(label, 0)
        if cap is not None and count_so_far >= cap:
            continue  # cap을 채운 클래스는 건너뛴다
        per_class[label] = count_so_far + 1

        images.append(load_image(path))
        labels.append(label)
        paths.append(str(path))

    return images, labels, paths


def embed_split(
    split_dir: Path | str,
    out: Path | str,
    cap: Optional[int] = None,
    batch_size: int = 32,
) -> Tuple[int, tuple]:
    """Stage 2 실행: crop 완료 split → embedding npz.

    batch_size는 VRAM에 맞춰 조절한다. 518px 입력에서는 배치가 커지면
    attention 행렬이 급격히 커지므로, VRAM이 작은 GPU에서는 8~16이 안전하다.
    """
    import numpy as np

    from encoder import BreedEncoder

    images, labels, paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    embeddings = np.asarray(
        BreedEncoder().encode_batch(images, batch_size=batch_size)
    )

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

    # 견종 이름을 가나다순으로 정렬해 저장 — npz 안에서 순서가 항상 같도록
    breeds = sorted(prototypes.keys())
    prototype_rows = []
    for breed in breeds:
        prototype_rows.append(np.asarray(prototypes[breed]))

    np.savez_compressed(
        out,
        breeds=np.asarray(breeds),
        prototypes=np.stack(prototype_rows),
        global_mean=np.asarray(result["global_mean"]),
    )
    return breeds


def load_prototypes(path: Path | str) -> dict:
    """prototype npz → scoring 모듈이 사용하는 dict."""
    import numpy as np

    data = np.load(path, allow_pickle=True)

    # npz의 breeds 배열(numpy 문자열)을 평범한 파이썬 문자열 리스트로 변환
    breeds = []
    for breed in data["breeds"]:
        breeds.append(str(breed))

    # {견종 이름: prototype 벡터} 딕셔너리로 복원
    prototypes = {}
    for index, breed in enumerate(breeds):
        prototypes[breed] = data["prototypes"][index]

    return {
        "prototypes": prototypes,
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

    # 주입받은 모델이 없으면 새로 만든다 (반복 호출 시엔 주입해서 재로딩 방지)
    if detector is None:
        detector = DogDetector()
    if encoder is None:
        encoder = BreedEncoder()

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
