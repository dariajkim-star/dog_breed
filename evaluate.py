"""평가지표 계산.

README 7 원칙:
- 견종 판별력과 OOD 거절력을 별도 평가
- 모든 튜닝은 val
- test는 최종 1회
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pipeline import load_prototypes, load_split_arrays


def evaluate_split(
    split_dir: Path | str,
    prototypes_path: Path | str,
    cap: Optional[int] = None,
) -> Dict[str, float]:
    """Top-1/Top-3 accuracy + purebred sanity check."""
    from encoder import BreedEncoder
    from scoring import similarity_scores

    prototypes = load_prototypes(prototypes_path)
    images, labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    encoder = BreedEncoder()
    embeddings = encoder.encode_batch(images)

    top1_hit = 0
    top3_hit = 0
    n = len(labels)

    for embedding, label in zip(embeddings, labels):
        scores = similarity_scores(embedding, prototypes)
        ranked = sorted(scores, key=scores.get, reverse=True)
        top1_hit += int(ranked[0] == label)
        top3_hit += int(label in ranked[:3])

    return {
        "n": n,
        "top1": top1_hit / n,
        "top3": top3_hit / n,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "purebred_sanity": top1_hit / n,
    }


def auroc_by_rank(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann–Whitney U 기반 AUROC. tie는 0.5 처리."""
    if not pos or not neg:
        raise ValueError("AUROC 계산에는 positive/negative 샘플이 각각 1개 이상 필요합니다.")

    scores = [(score, 1) for score in pos] + [(score, 0) for score in neg]
    scores.sort(key=lambda item: item[0])

    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and scores[j][0] == scores[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_pos, n_neg = len(pos), len(neg)
    rank_sum_pos = sum(
        rank
        for rank, (_score, label) in zip(ranks, scores)
        if label == 1
    )
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def max_sims_for_dir(
    split_dir: Path | str,
    proto: dict,
    cap: Optional[int] = None,
    *,
    encoder=None,
) -> List[float]:
    """디렉터리 이미지별 max prototype similarity."""
    from encoder import BreedEncoder
    from scoring import similarity_scores

    images, _labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    encoder = encoder or BreedEncoder()
    embeddings = encoder.encode_batch(images)
    return [
        max(similarity_scores(embedding, proto).values())
        for embedding in embeddings
    ]


def evaluate_ood(
    dog_dir: Path | str,
    cat_dir: Path | str,
    prototypes_path: Path | str,
    cap: Optional[int] = None,
) -> Dict[str, float]:
    """개(in-distribution) vs 고양이(OOD) max-similarity AUROC."""
    from encoder import BreedEncoder

    proto = load_prototypes(prototypes_path)

    # 기존 코드는 dog/cat 평가 때 DINOv2를 두 번 생성했다.
    # 하나의 encoder를 재사용해 로딩 시간과 GPU 메모리 낭비를 줄인다.
    encoder = BreedEncoder()
    dog_sims = max_sims_for_dir(dog_dir, proto, cap=cap, encoder=encoder)
    cat_sims = max_sims_for_dir(cat_dir, proto, cap=cap, encoder=encoder)

    return {
        "n_dog": len(dog_sims),
        "n_cat": len(cat_sims),
        "auroc": auroc_by_rank(dog_sims, cat_sims),
    }
