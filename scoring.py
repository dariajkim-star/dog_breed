# -*- coding: utf-8 -*-
"""Stage 4~5 — Prototype Similarity, Unknown/OOD, Calibration.

모델 추론 결과를 phenotype similarity score로 변환하는 순수 계산 로직만 둔다.
"""

from __future__ import annotations

import numpy as np


def _l2norm(vector: np.ndarray) -> np.ndarray:
    """벡터 길이를 1로 만든다 (L2 정규화).

    max(..., 1e-12)는 벡터가 0일 때 0으로 나누는 사고를 막기 위한 안전장치.
    """
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-12)


def similarity_scores(embedding: np.ndarray, prototype_data: dict) -> dict[str, float]:
    """query embedding과 견종별 prototype 간 cosine similarity를 계산한다.

    prototype을 만들 때 global_mean을 뺐으므로(CL2N centering),
    query 쪽에도 똑같이 빼 줘야 공정한 비교가 된다.
    """
    centered = np.asarray(embedding, dtype=np.float32) - prototype_data["global_mean"]
    query = _l2norm(centered)

    # 두 벡터 모두 길이 1이므로 내적(dot product)이 곧 cosine similarity
    scores: dict[str, float] = {}
    for breed, prototype in prototype_data["prototypes"].items():
        scores[breed] = float(np.dot(query, prototype))
    return scores


def predict(
    embedding: np.ndarray,
    prototype_data: dict,
    threshold: float,
    top_k: int = 3,
    temperature: float = 0.1,
) -> dict:
    """Unknown 분기 후 Top-K phenotype similarity score를 반환한다 (README 4~5).

    temperature(기본 0.1)는 softmax를 얼마나 뾰족하게 만들지 정하는 값 —
    작을수록 1등 견종에 확률이 몰린다. 최종값은 val에서 조정한다.
    """
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if temperature <= 0:
        raise ValueError("temperature는 0보다 커야 합니다.")

    similarities = similarity_scores(embedding, prototype_data)

    # dict를 (견종 리스트, 점수 배열) 두 개로 나눠 담는다
    breeds: list[str] = []
    score_list: list[float] = []
    for breed, score in similarities.items():
        breeds.append(breed)
        score_list.append(score)
    values = np.array(score_list, dtype=np.float32)

    if len(values) == 0:
        # prototype이 하나도 없으면 비교 자체가 불가능 → 항상 Unknown
        return {"unknown": True, "max_sim": float("-inf"), "topk": []}
    max_sim = float(values.max())

    # Unknown/OOD 분기 (README 4 필수) — 가장 닮은 견종조차 threshold 미만이면 거절.
    # 단, 순위 계산은 그대로 진행해 topk를 함께 돌려준다.
    #
    # threshold는 순종 val 분포로 잡혀 있어(순종 95%가 통과하는 지점) 믹스견은
    # 구조적으로 이 선 아래에 떨어진다 — 순종 prototype과 원래 덜 닮기 때문이다.
    # 여기서 topk를 비워 버리면 정작 이 프로젝트의 주 대상인 믹스견에게
    # 아무 정보도 못 주게 된다. 거절 여부(unknown)와 순위(topk)를 분리해서
    # 표시 방법은 호출부가 정하도록 한다.
    unknown = max_sim < threshold

    # temperature softmax (README 5 calibration)
    logits = values / temperature
    logits = logits - logits.max()  # 가장 큰 값을 0으로 — exp() overflow 방지용 표준 기법
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum()

    # 확률 내림차순 상위 top_k개 (argsort는 오름차순이라 부호를 뒤집어 내림차순으로)
    order = np.argsort(-probabilities)
    topk: list[tuple[str, float]] = []
    for index in order[:top_k]:
        breed = breeds[index]
        percent = float(probabilities[index] * 100.0)
        topk.append((breed, percent))

    return {"unknown": unknown, "max_sim": max_sim, "topk": topk}
