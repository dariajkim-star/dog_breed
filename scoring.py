# -*- coding: utf-8 -*-
"""Stage 4~5 — Prototype Similarity, Unknown/OOD, Calibration.

모델 추론 결과를 phenotype similarity score로 변환하는 순수 계산 로직만 둔다.
"""

from __future__ import annotations

import numpy as np


def _l2norm(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def similarity_scores(emb: np.ndarray, proto: dict) -> dict[str, float]:
    """query embedding과 breed prototype 간 cosine similarity를 계산한다."""
    query = _l2norm(np.asarray(emb, dtype=np.float32) - proto["global_mean"])
    return {
        breed: float(query @ prototype)
        for breed, prototype in proto["prototypes"].items()
    }


def predict(
    emb: np.ndarray,
    proto: dict,
    threshold: float,
    top_k: int = 3,
    temperature: float = 0.1,
) -> dict:
    """Unknown 분기 후 Top-K phenotype similarity score를 반환한다."""
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if temperature <= 0:
        raise ValueError("temperature는 0보다 커야 합니다.")

    similarities = similarity_scores(emb, proto)
    breeds = list(similarities)
    values = np.array([similarities[breed] for breed in breeds], dtype=np.float32)
    max_sim = float(values.max()) if len(values) else float("-inf")

    if max_sim < threshold:
        return {"unknown": True, "max_sim": max_sim, "topk": []}

    logits = values / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    order = np.argsort(-probabilities)[:top_k]
    topk = [(breeds[i], float(probabilities[i] * 100.0)) for i in order]
    return {"unknown": False, "max_sim": max_sim, "topk": topk}
