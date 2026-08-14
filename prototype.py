# -*- coding: utf-8 -*-
"""Stage 3 — Breed Prototype Generation.

클래스별 embedding을 대표 prototype으로 축약한다.
README의 cap, outlier 제거, CL2N centering 규칙을 구현한다.
"""

from __future__ import annotations

import numpy as np


def _l2norm(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def build_prototypes(
    embs_by_class: dict[str, np.ndarray],
    cap: int = 50,
    outlier_frac: float = 0.05,
) -> dict:
    """견종별 embedding으로 prototype과 global mean을 생성한다."""
    if not embs_by_class:
        raise ValueError("prototype을 만들 embedding이 없습니다.")
    if cap <= 0:
        raise ValueError("cap은 1 이상이어야 합니다.")
    if not 0.0 <= outlier_frac < 1.0:
        raise ValueError("outlier_frac은 0 이상 1 미만이어야 합니다.")

    normed: dict[str, np.ndarray] = {}
    for breed, embeddings in embs_by_class.items():
        emb = np.asarray(embeddings, dtype=np.float32)[:cap]
        if len(emb) == 0:
            continue
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        normed[breed] = emb

    if not normed:
        raise ValueError("prototype을 만들 유효한 embedding이 없습니다.")

    # 기존 구현과 동일하게 outlier 제거 전 전체 embedding 평균을 CL2N global mean으로 사용한다.
    global_mean = np.concatenate(list(normed.values()), axis=0).mean(axis=0).astype(np.float32)

    def make_prototype(embeddings: np.ndarray) -> np.ndarray:
        centroid = _l2norm(embeddings.mean(axis=0))
        return _l2norm(centroid - global_mean).astype(np.float32)

    prototypes: dict[str, np.ndarray] = {}
    for breed, embeddings in normed.items():
        centroid = _l2norm(embeddings.mean(axis=0))
        n_drop = int(len(embeddings) * outlier_frac)
        if n_drop > 0 and len(embeddings) - n_drop >= 1:
            cosine = embeddings @ centroid
            keep = np.argsort(cosine)[n_drop:]
            embeddings = embeddings[keep]
        prototypes[breed] = make_prototype(embeddings)

    return {
        "prototypes": prototypes,
        "global_mean": global_mean,
    }
