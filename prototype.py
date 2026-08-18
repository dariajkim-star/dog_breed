# -*- coding: utf-8 -*-
"""Stage 3 — Breed Prototype Generation.

클래스별 embedding을 대표 prototype으로 축약한다.
README 3의 규칙: 클래스당 cap장 → 정규화 → 평균 → 재정규화,
outlier 하위 5% 제거 후 재계산, CL2N centering.
"""

from __future__ import annotations

import numpy as np


def _l2norm(vector: np.ndarray) -> np.ndarray:
    """벡터 길이를 1로 만든다 (L2 정규화).

    max(..., 1e-12)는 벡터가 0일 때 0으로 나누는 사고를 막기 위한 안전장치.
    """
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-12)


def build_prototypes(
    embs_by_class: dict[str, np.ndarray],
    cap: int = 50,
    outlier_frac: float = 0.05,
) -> dict:
    """견종별 embedding 묶음으로 prototype과 global mean을 생성한다.

    반환: {"prototypes": {견종: (384,) 벡터}, "global_mean": (384,) 벡터}
    """
    if not embs_by_class:
        raise ValueError("prototype을 만들 embedding이 없습니다.")
    if cap <= 0:
        raise ValueError("cap은 1 이상이어야 합니다.")
    if not 0.0 <= outlier_frac < 1.0:
        raise ValueError("outlier_frac은 0 이상 1 미만이어야 합니다.")

    # 1단계: 클래스마다 cap장까지만 쓰고, 각 embedding을 길이 1로 정규화
    #        (README 3: 20~50장이면 수렴하므로 데이터가 많아도 cap으로 자른다)
    normed: dict[str, np.ndarray] = {}
    for breed, embeddings in embs_by_class.items():
        capped = np.asarray(embeddings, dtype=np.float32)[:cap]
        if len(capped) == 0:
            continue
        lengths = np.linalg.norm(capped, axis=1, keepdims=True)
        capped = capped / np.maximum(lengths, 1e-12)  # 0 나누기 방지
        normed[breed] = capped

    if not normed:
        raise ValueError("prototype을 만들 유효한 embedding이 없습니다.")

    # 2단계: 전체 embedding의 평균 = CL2N centering용 global mean
    #        (모든 견종이 공유하는 "평균적인 개다움"을 빼서 견종별 차이만 남기는 기법)
    all_embeddings = np.concatenate(list(normed.values()), axis=0)
    global_mean = all_embeddings.mean(axis=0).astype(np.float32)

    # 3단계: 견종별 prototype 만들기
    prototypes: dict[str, np.ndarray] = {}
    for breed, embeddings in normed.items():
        # (a) 1차 중심점: 평균 → 재정규화
        centroid = _l2norm(embeddings.mean(axis=0))

        # (b) outlier 제거: 중심점과 덜 닮은 하위 outlier_frac(기본 5%)을 버린다
        #     (이상한 사진 — 뒷모습, 가려짐 등 — 이 평균을 오염시키지 않도록)
        drop_count = int(len(embeddings) * outlier_frac)
        if drop_count > 0 and len(embeddings) - drop_count >= 1:
            cosine_to_centroid = embeddings @ centroid
            keep_indices = np.argsort(cosine_to_centroid)[drop_count:]
            embeddings = embeddings[keep_indices]
            # outlier를 뺐으니 중심점을 다시 계산
            centroid = _l2norm(embeddings.mean(axis=0))

        # (c) CL2N: global mean을 빼고 재정규화한 것이 최종 prototype
        prototypes[breed] = _l2norm(centroid - global_mean).astype(np.float32)

    return {
        "prototypes": prototypes,
        "global_mean": global_mean,
    }
