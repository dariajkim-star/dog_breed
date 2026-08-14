"""CLI command handlers.

출력 포맷만 책임지고 실제 계산은 stage/pipeline/evaluate 모듈에 위임한다.
"""

from __future__ import annotations

import argparse

from constants import UNKNOWN_MESSAGE


def detect(args: argparse.Namespace) -> None:
    from detection import DogDetector
    from preprocessing import load_image

    image = load_image(args.image)
    detections = DogDetector(weights=args.weights, conf=args.conf).detect(image)

    if not detections:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return

    print(f"탐지 결과: {len(detections)}건")
    for index, (bbox, confidence) in enumerate(detections, start=1):
        x1, y1, x2, y2 = bbox
        print(
            f"  [{index}] bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  "
            f"conf={confidence:.3f}"
        )


def embed(args: argparse.Namespace) -> None:
    from pipeline import embed_split

    print(f"split 순회 중: {args.split_dir}")
    n, shape = embed_split(args.split_dir, args.out, cap=args.cap)
    print(f"저장 완료: {args.out}  ({n}장, embeddings={shape})")


def prototype(args: argparse.Namespace) -> None:
    from pipeline import build_and_save_prototypes

    breeds = build_and_save_prototypes(args.embeddings, args.out, cap=args.cap)
    print(f"저장 완료: {args.out}  ({len(breeds)}종)")


def infer(args: argparse.Namespace) -> None:
    from pipeline import infer_image

    result = infer_image(
        args.image,
        args.prototypes,
        threshold=args.threshold,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    if result is None:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return

    print(f"입력: {args.image}")
    print(f"max similarity = {result['max_sim']:.3f} (threshold={args.threshold})")
    if result["unknown"]:
        print(f"⚠️ Unknown: {UNKNOWN_MESSAGE}")
        return

    print("Phenotype Similarity Score (DNA 혈통 비율 아님):")
    for breed, pct in result["topk"]:
        print(f"  {breed:<25s} {pct:5.1f}%")


def evaluate_breed(args: argparse.Namespace) -> None:
    from evaluate import evaluate_split

    print(f"평가 시작: {args.split_dir}")
    result = evaluate_split(args.split_dir, args.prototypes, cap=args.cap)
    print("=== 평가 결과 (README 7. 평가지표) ===")
    print(
        f"Top-1 accuracy            : {result['top1']:.4f}  "
        f"({result['top1_hit']}/{result['n']})"
    )
    print(
        f"Top-3 accuracy            : {result['top3']:.4f}  "
        f"({result['top3_hit']}/{result['n']})"
    )
    print(f"순종 sanity check (Top-1) : {result['purebred_sanity']:.4f}")


def evaluate_ood(args: argparse.Namespace) -> None:
    from evaluate import evaluate_ood as run_evaluate_ood

    print(f"개(in-dist): {args.dog_dir} / 고양이(OOD): {args.cat_dir}")
    result = run_evaluate_ood(
        args.dog_dir,
        args.cat_dir,
        args.prototypes,
        cap=args.cap,
    )
    print("=== OOD 평가 결과 (README 7. 평가지표) ===")
    print(f"개 {result['n_dog']}장 / 고양이 {result['n_cat']}장")
    print(f"max-similarity AUROC : {result['auroc']:.4f}")
