"""평가지표 계산 로직 — CLI(main.py)와 분리 (README 7. 평가지표).

    evaluate_split : Top-1/Top-3 accuracy + 순종 sanity check
    evaluate_ood   : 개 vs 고양이 max-similarity AUROC
    auroc_by_rank  : sklearn 의존성 없는 rank 기반 AUROC

원칙: 기능별로 따로 평가하고 한 숫자로 뭉치지 않는다. 모든 튜닝은 val에서,
test는 최종 1회만 (README 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pipeline import load_prototypes, load_split_arrays


def evaluate_split(
    split_dir: Path | str, prototypes_path: Path | str, cap: Optional[int] = None
) -> Dict[str, float]:
    """split 전체에 대해 Top-1/Top-3 accuracy를 계산한다.

    순종 sanity check(README 7 메인 KPI) = 순종 test 이미지의 Top-1 정답률.
    breed_body split은 전부 순종이므로 Top-1 accuracy와 동일 값.
    """
    from model import BreedEncoder, similarity_scores

    proto = load_prototypes(prototypes_path)
    imgs, labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not imgs:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    embeddings = BreedEncoder().encode_batch(imgs)

    top1_hit = 0
    top3_hit = 0
    n = len(labels)
    for emb, label in zip(embeddings, labels):
        scores = similarity_scores(emb, proto)
        ranked = sorted(scores, key=scores.get, reverse=True)
        if ranked[0] == label:
            top1_hit += 1
        if label in ranked[:3]:
            top3_hit += 1

    return {
        "n": n,
        "top1": top1_hit / n,
        "top3": top3_hit / n,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "purebred_sanity": top1_hit / n,
    }


def auroc_by_rank(pos: Sequence[float], neg: Sequence[float]) -> float:
    """rank 기반 AUROC (Mann–Whitney U) — sklearn 의존성 없이 계산.

    pos(개) 점수가 neg(고양이) 점수보다 클 확률. 동점은 0.5 처리.
    """
    scores = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    scores.sort(key=lambda t: t[0])

    # 동점 그룹에 평균 rank 부여
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and scores[j][0] == scores[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based 평균 rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_pos, n_neg = len(pos), len(neg)
    rank_sum_pos = sum(r for r, (_s, y) in zip(ranks, scores) if y == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def max_sims_for_dir(
    split_dir: Path | str, proto: dict, cap: Optional[int] = None
) -> List[float]:
    """디렉토리 이미지들의 prototype 대비 max cosine similarity 목록."""
    from model import BreedEncoder, similarity_scores

    imgs, _labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not imgs:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")
    embeddings = BreedEncoder().encode_batch(imgs)
    return [max(similarity_scores(emb, proto).values()) for emb in embeddings]


def evaluate_ood(
    dog_dir: Path | str,
    cat_dir: Path | str,
    prototypes_path: Path | str,
    cap: Optional[int] = None,
) -> Dict[str, float]:
    """개(in-dist) vs 고양이(OOD)의 max-similarity 분포로 AUROC 계산."""
    proto = load_prototypes(prototypes_path)
    dog_sims = max_sims_for_dir(dog_dir, proto, cap=cap)
    cat_sims = max_sims_for_dir(cat_dir, proto, cap=cap)
    return {
        "n_dog": len(dog_sims),
        "n_cat": len(cat_sims),
        "auroc": auroc_by_rank(dog_sims, cat_sims),
    }
