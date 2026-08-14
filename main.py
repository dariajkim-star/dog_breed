"""FindDogBreed 파이프라인 CLI — 단계별 실행 엔트리포인트.

각 서브커맨드는 파이프라인의 한 단계만 수행한다 (README 아키텍처 참조).
    detect    : 사진에서 강아지 bbox 탐지 (Stage 1)
    embed     : 이미 crop된 split 이미지 → embedding npz 저장 (Stage 2)
    prototype : embedding npz → 25종 prototype npz 생성 (Stage 3)
    infer     : 단일 이미지 전체 경로 (detect → crop → encode → predict)
    eval      : Top-1/Top-3 + 순종 sanity check (README 7. 평가지표)
    eval-ood  : 개 vs 고양이 max-similarity AUROC (README 7. 평가지표)

무거운 torch 계열 import(model, preprocessing)는 각 함수 내부에서 lazy import —
`python main.py --help`가 빠르게 뜨도록 유지.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Unknown 출력 문구 — README 4. Inference 참조
UNKNOWN_MESSAGE = "현재 지원하는 품종만으로 설명하기 어렵습니다"


# ---------------------------------------------------------------------------
# 서브커맨드 구현
# ---------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> None:
    """detect: 사진 한 장에서 강아지 bbox/confidence 출력 (Stage 1)."""
    from preprocessing import load_image
    from model import DogDetector

    img = load_image(args.image)
    detector = DogDetector(weights=args.weights, conf=args.conf)
    detections = detector.detect(img)

    if not detections:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return

    print(f"탐지 결과: {len(detections)}건")
    for i, (bbox, conf) in enumerate(detections, start=1):
        x1, y1, x2, y2 = bbox
        print(f"  [{i}] bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  conf={conf:.3f}")


def _iter_split_arrays(
    split_dir: Path, cap: int | None = None
) -> Tuple[List, List[str], List[str]]:
    """split 디렉토리를 순회해 (이미지 리스트, 라벨, 경로 문자열)을 모은다.

    processed 데이터는 이미 518px 정사각 crop이므로 detection 불필요
    (docs/handover.md 기능 2 참조). cap 지정 시 클래스당 앞에서부터 cap장만
    사용 (CPU 테스트 런 용도 — prototype은 어차피 클래스당 ≤50장).
    """
    from preprocessing import load_image, iter_split

    imgs, labels, paths = [], [], []
    per_class: Dict[str, int] = {}
    for path, label in iter_split(split_dir):
        if cap is not None and per_class.get(label, 0) >= cap:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        imgs.append(load_image(path))
        labels.append(label)
        paths.append(str(path))
    return imgs, labels, paths


def cmd_embed(args: argparse.Namespace) -> None:
    """embed: crop 완료된 split 이미지 전체를 embedding으로 변환해 npz 저장 (Stage 2)."""
    import numpy as np
    from model import BreedEncoder

    split_dir = Path(args.split_dir)
    print(f"split 순회 중: {split_dir}")
    imgs, labels, paths = _iter_split_arrays(split_dir, cap=getattr(args, 'cap', None))
    if not imgs:
        raise SystemExit(f"이미지를 찾지 못했습니다: {split_dir}")
    print(f"{len(imgs)}장 로드 완료 — embedding 추출 시작")

    encoder = BreedEncoder()
    embeddings = encoder.encode_batch(imgs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=np.asarray(embeddings),
        labels=np.asarray(labels),
        paths=np.asarray(paths),
    )
    print(f"저장 완료: {out}  (embeddings={np.asarray(embeddings).shape})")


def cmd_prototype(args: argparse.Namespace) -> None:
    """prototype: embedding npz → 견종별 prototype 생성 후 npz 저장 (Stage 3)."""
    import numpy as np
    from model import build_prototypes

    data = np.load(args.embeddings, allow_pickle=True)
    embeddings, labels = data["embeddings"], data["labels"]

    # 견종별로 embedding을 모은다 (클래스당 cap장 캡 — README 3 참조)
    embs_by_class: Dict[str, List] = {}
    for emb, label in zip(embeddings, labels):
        embs_by_class.setdefault(str(label), []).append(emb)
    print(f"클래스 {len(embs_by_class)}종, 총 {len(labels)}장으로 prototype 구축")

    result = build_prototypes(embs_by_class, cap=args.cap)
    protos = result["prototypes"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    breeds = sorted(protos.keys())
    np.savez_compressed(
        out,
        breeds=np.asarray(breeds),
        prototypes=np.stack([np.asarray(protos[b]) for b in breeds]),
        global_mean=np.asarray(result["global_mean"]),
    )
    print(f"저장 완료: {out}  ({len(breeds)}종)")


def _load_prototypes(path: str) -> dict:
    """prototype npz를 model 함수들이 기대하는 dict 형태로 복원한다."""
    import numpy as np

    data = np.load(path, allow_pickle=True)
    breeds = [str(b) for b in data["breeds"]]
    return {
        "prototypes": {b: data["prototypes"][i] for i, b in enumerate(breeds)},
        "global_mean": data["global_mean"],
    }


def _embed_single_image(image_path: str) -> "object":
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


def cmd_infer(args: argparse.Namespace) -> None:
    """infer: 단일 이미지 전체 경로 실행 후 결과 pretty-print (README 4)."""
    from model import predict

    emb = _embed_single_image(args.image)
    if emb is None:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return

    proto = _load_prototypes(args.prototypes)
    result = predict(emb, proto, threshold=args.threshold,
                     top_k=args.top_k, temperature=args.temperature)

    print(f"입력: {args.image}")
    print(f"max similarity = {result['max_sim']:.3f} (threshold={args.threshold})")
    if result["unknown"]:
        # Unknown/OOD 분기 — README 4 필수 문구
        print(f"⚠️ Unknown: {UNKNOWN_MESSAGE}")
        return
    print("Phenotype Similarity Score (DNA 혈통 비율 아님):")
    for breed, pct in result["topk"]:
        print(f"  {breed:<25s} {pct:5.1f}%")


def cmd_eval(args: argparse.Namespace) -> None:
    """eval: test split에서 Top-1/Top-3 accuracy + 순종 sanity check (README 7)."""
    import numpy as np
    from model import BreedEncoder, similarity_scores

    proto = _load_prototypes(args.prototypes)
    split_dir = Path(args.split_dir)
    imgs, labels, _paths = _iter_split_arrays(split_dir, cap=getattr(args, 'cap', None))
    if not imgs:
        raise SystemExit(f"이미지를 찾지 못했습니다: {split_dir}")
    print(f"{len(imgs)}장 평가 시작: {split_dir}")

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

    # README 7. 평가지표 — 순종 sanity check = 순종 이미지 Top-1 정답률
    print("=== 평가 결과 (README 7. 평가지표) ===")
    print(f"Top-1 accuracy            : {top1_hit / n:.4f}  ({top1_hit}/{n})")
    print(f"Top-3 accuracy            : {top3_hit / n:.4f}  ({top3_hit}/{n})")
    print(f"순종 sanity check (Top-1) : {top1_hit / n:.4f}")


def _auroc_by_rank(pos: Sequence[float], neg: Sequence[float]) -> float:
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


def _max_sims_for_dir(split_dir: Path, proto: dict) -> List[float]:
    """디렉토리 이미지들의 25종 prototype 대비 max cosine similarity 목록."""
    from model import BreedEncoder, similarity_scores

    imgs, _labels, _paths = _iter_split_arrays(split_dir, cap=getattr(args, 'cap', None))
    if not imgs:
        raise SystemExit(f"이미지를 찾지 못했습니다: {split_dir}")
    embeddings = BreedEncoder().encode_batch(imgs)
    return [max(similarity_scores(emb, proto).values()) for emb in embeddings]


def cmd_eval_ood(args: argparse.Namespace) -> None:
    """eval-ood: 개 vs 고양이 max-similarity 분포로 AUROC 계산 (README 7, OOD)."""
    proto = _load_prototypes(args.prototypes)

    print(f"개(in-dist) 디렉토리 처리: {args.dog_dir}")
    dog_sims = _max_sims_for_dir(Path(args.dog_dir), proto)
    print(f"고양이(OOD) 디렉토리 처리: {args.cat_dir}")
    cat_sims = _max_sims_for_dir(Path(args.cat_dir), proto)

    auroc = _auroc_by_rank(dog_sims, cat_sims)
    print("=== OOD 평가 결과 (README 7. 평가지표) ===")
    print(f"개 {len(dog_sims)}장 / 고양이 {len(cat_sims)}장")
    print(f"max-similarity AUROC : {auroc:.4f}")


# ---------------------------------------------------------------------------
# argparse 구성
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FindDogBreed 파이프라인 — 단계별 실행 (한 번에 한 단계만)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    p = sub.add_parser("detect", help="사진에서 강아지 bbox 탐지 (Stage 1)")
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO 가중치 (기본 yolo11n.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence 임계값")
    p.set_defaults(func=cmd_detect)

    # embed
    p = sub.add_parser("embed", help="crop 완료 split → embedding npz 저장 (Stage 2)")
    p.add_argument("--split-dir", default="data/processed/breed_body/train",
                   help="이미 crop된 split 디렉토리")
    p.add_argument("--out", default="artifacts/embeddings_train.npz", help="출력 npz 경로")
    p.add_argument("--cap", type=int, default=None,
                   help="클래스당 최대 장수 (CPU 테스트 런 용도)")
    p.set_defaults(func=cmd_embed)

    # prototype
    p = sub.add_parser("prototype", help="embedding npz → 견종 prototype 생성 (Stage 3)")
    p.add_argument("--embeddings", default="artifacts/embeddings_train.npz",
                   help="embed 단계 출력 npz")
    p.add_argument("--out", default="artifacts/prototypes.npz", help="출력 npz 경로")
    p.add_argument("--cap", type=int, default=50, help="클래스당 이미지 캡 (README 3)")
    p.set_defaults(func=cmd_prototype)

    # infer
    p = sub.add_parser("infer", help="단일 이미지 추론 (detect→crop→encode→predict)")
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Unknown 분기 threshold (val에서 결정)")
    p.add_argument("--top-k", type=int, default=3, help="상위 K개 견종 출력")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="calibration temperature (README 5)")
    p.set_defaults(func=cmd_infer)

    # eval
    p = sub.add_parser("eval", help="Top-1/Top-3 + 순종 sanity check (README 7)")
    p.add_argument("--split-dir", default="data/processed/breed_body/test",
                   help="평가 split 디렉토리 (test는 최종 1회만!)")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    p.add_argument("--cap", type=int, default=None,
                   help="클래스당 최대 장수 (CPU 테스트 런 용도)")
    p.set_defaults(func=cmd_eval)

    # eval-ood
    p = sub.add_parser("eval-ood", help="개 vs 고양이 max-sim AUROC (README 7, OOD)")
    p.add_argument("--dog-dir", default="data/processed/breed_body/test",
                   help="in-distribution 개 이미지 디렉토리")
    p.add_argument("--cat-dir", default="data/processed/ood/test",
                   help="OOD 고양이 이미지 디렉토리")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    p.set_defaults(func=cmd_eval_ood)

    return parser


def main() -> None:
    # Windows 콘솔(cp949)에서 한글/특수문자 출력 깨짐 방지
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
