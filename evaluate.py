"""평가지표 계산.

README 7 원칙:
- 견종 판별력과 OOD 거절력을 별도 평가 (한 숫자로 뭉치지 않음)
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
    *,
    encoder=None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Top-1/Top-3 accuracy + purebred sanity check.

    encoder를 넘기면 DINOv2를 다시 로드하지 않는다 (threshold 스윕 등 반복 평가용).

    purebred_sanity가 top1과 같은 값인 이유:
    breed_body split은 전부 순종 이미지라서 "순종 sanity check(순종 사진의 Top-1
    정답률)" = "이 split의 Top-1 accuracy"가 정의상 동일하다. 믹스견 이미지가
    섞인 평가셋이 생기면 순종만 골라 따로 계산하도록 바꿔야 한다.
    """
    from encoder import BreedEncoder
    from scoring import similarity_scores

    prototype_data = load_prototypes(prototypes_path)
    images, labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    if encoder is None:
        encoder = BreedEncoder()
    embeddings = encoder.encode_batch(images, batch_size=batch_size)

    top1_hit = 0
    top3_hit = 0
    total = len(labels)

    for embedding, correct_label in zip(embeddings, labels):
        scores = similarity_scores(embedding, prototype_data)

        # 점수 내림차순으로 견종 순위 만들기
        # (정렬을 위해 (점수, 견종) 짝을 만들고, 큰 점수부터 정렬)
        score_breed_pairs: List[tuple] = []
        for breed, score in scores.items():
            score_breed_pairs.append((score, breed))
        score_breed_pairs.sort(reverse=True)

        ranked_breeds: List[str] = []
        for _score, breed in score_breed_pairs:
            ranked_breeds.append(breed)

        # 1등이 정답이면 Top-1 적중, 3등 안에 정답이 있으면 Top-3 적중
        if ranked_breeds[0] == correct_label:
            top1_hit = top1_hit + 1
        if correct_label in ranked_breeds[:3]:
            top3_hit = top3_hit + 1

    return {
        "n": total,
        "top1": top1_hit / total,
        "top3": top3_hit / total,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        # breed_body는 전부 순종이므로 top1과 동일 (docstring 참조)
        "purebred_sanity": top1_hit / total,
    }


def _assign_average_ranks(sorted_scores: List[tuple]) -> List[float]:
    """오름차순 정렬된 (점수, 라벨) 리스트에 1등부터 순위를 매긴다.

    동점(같은 점수)이 여러 개면 그 구간의 평균 순위를 나눠 갖는다.
    예: 2, 3, 4등이 동점이면 셋 다 3.0등.
    """
    ranks = [0.0] * len(sorted_scores)
    start = 0
    while start < len(sorted_scores):
        # start와 점수가 같은 구간 [start, end)를 찾는다
        end = start
        while end < len(sorted_scores) and sorted_scores[end][0] == sorted_scores[start][0]:
            end = end + 1
        # 이 구간의 평균 순위 (순위는 1부터 시작)
        average_rank = (start + end + 1) / 2.0
        for index in range(start, end):
            ranks[index] = average_rank
        start = end
    return ranks


def auroc_by_rank(pos: Sequence[float], neg: Sequence[float]) -> float:
    """AUROC를 순위 기반(Mann-Whitney U 통계량)으로 계산한다. sklearn 불필요.

    의미: "개 점수 하나와 고양이 점수 하나를 무작위로 뽑았을 때,
    개 점수가 더 클 확률". 1.0이면 완벽 구분, 0.5면 동전 던지기 수준.
    공식 참고: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    """
    if not pos or not neg:
        raise ValueError("AUROC 계산에는 positive/negative 샘플이 각각 1개 이상 필요합니다.")

    # (점수, 라벨) 짝을 만든다 — 라벨 1 = 개(positive), 0 = 고양이(negative)
    scores: List[tuple] = []
    for score in pos:
        scores.append((score, 1))
    for score in neg:
        scores.append((score, 0))
    scores.sort()  # 튜플은 첫 요소(점수) 기준으로 오름차순 정렬된다

    ranks = _assign_average_ranks(scores)

    # 개(positive) 점수들의 순위 합계
    rank_sum_pos = 0.0
    for rank, (_score, label) in zip(ranks, scores):
        if label == 1:
            rank_sum_pos = rank_sum_pos + rank

    n_pos = len(pos)
    n_neg = len(neg)
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


def max_sims_for_dir(
    split_dir: Path | str,
    prototype_data: dict,
    cap: Optional[int] = None,
    *,
    encoder=None,
    batch_size: int = 32,
) -> List[float]:
    """디렉터리의 각 이미지가 prototype들과 얼마나 닮았는지, 최고 유사도만 모은다."""
    from encoder import BreedEncoder
    from scoring import similarity_scores

    images, _labels, _paths = load_split_arrays(split_dir, cap=cap)
    if not images:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {split_dir}")

    if encoder is None:
        encoder = BreedEncoder()
    embeddings = encoder.encode_batch(images, batch_size=batch_size)

    max_similarities: List[float] = []
    for embedding in embeddings:
        scores = similarity_scores(embedding, prototype_data)
        best_score = max(scores.values())
        max_similarities.append(best_score)
    return max_similarities


def evaluate_ood(
    dog_dir: Path | str,
    cat_dir: Path | str,
    prototypes_path: Path | str,
    cap: Optional[int] = None,
    *,
    encoder=None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """개(in-distribution) vs 고양이(OOD)의 max-similarity 분포로 AUROC 계산.

    encoder를 넘기면 DINOv2를 다시 로드하지 않는다.
    """
    from encoder import BreedEncoder

    prototype_data = load_prototypes(prototypes_path)

    # 개/고양이 평가에 같은 encoder를 재사용 — 모델을 두 번 로드하지 않는다
    if encoder is None:
        encoder = BreedEncoder()
    dog_sims = max_sims_for_dir(
        dog_dir, prototype_data, cap=cap, encoder=encoder, batch_size=batch_size
    )
    cat_sims = max_sims_for_dir(
        cat_dir, prototype_data, cap=cap, encoder=encoder, batch_size=batch_size
    )

    return {
        "n_dog": len(dog_sims),
        "n_cat": len(cat_sims),
        "auroc": auroc_by_rank(dog_sims, cat_sims),
    }
