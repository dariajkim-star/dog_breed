"""argparse schema.

main.py에서 CLI 정의를 분리해 엔트리포인트가 실행 제어만 담당하도록 한다.
"""

from __future__ import annotations

import argparse


def _add_cap(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="클래스당 최대 장수 (CPU 테스트 런 용도)",
    )


def build_parser() -> argparse.ArgumentParser:
    from commands import detect, embed, evaluate_breed, evaluate_ood, infer, prototype

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FindDogBreed 파이프라인 - 단계별 실행",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_parser = sub.add_parser("detect", help="사진에서 강아지 bbox 탐지 (Stage 1)")
    detect_parser.add_argument("--image", required=True, help="입력 이미지 경로")
    detect_parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="YOLO 가중치",
    )
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="detection confidence 임계값",
    )
    detect_parser.set_defaults(func=detect)

    embed_parser = sub.add_parser(
        "embed",
        help="crop 완료 split → embedding npz 저장 (Stage 2)",
    )
    embed_parser.add_argument(
        "--split-dir",
        default="data/processed/breed_body/train",
        help="이미 crop된 split 디렉터리",
    )
    embed_parser.add_argument(
        "--out",
        default="artifacts/embeddings_train.npz",
        help="출력 npz 경로",
    )
    _add_cap(embed_parser)
    embed_parser.set_defaults(func=embed)

    prototype_parser = sub.add_parser(
        "prototype",
        help="embedding npz → 견종 prototype 생성 (Stage 3)",
    )
    prototype_parser.add_argument(
        "--embeddings",
        default="artifacts/embeddings_train.npz",
        help="embed 단계 출력 npz",
    )
    prototype_parser.add_argument(
        "--out",
        default="artifacts/prototypes.npz",
        help="출력 npz 경로",
    )
    prototype_parser.add_argument(
        "--cap",
        type=int,
        default=50,
        help="클래스당 prototype 이미지 cap",
    )
    prototype_parser.set_defaults(func=prototype)

    infer_parser = sub.add_parser(
        "infer",
        help="단일 이미지 추론 (detect→crop→encode→predict)",
    )
    infer_parser.add_argument("--image", required=True, help="입력 이미지 경로")
    infer_parser.add_argument(
        "--prototypes",
        default="artifacts/prototypes.npz",
        help="prototype npz",
    )
    infer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Unknown threshold (validation에서 결정)",
    )
    infer_parser.add_argument("--top-k", type=int, default=3, help="상위 K개 견종 출력")
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature scaling 값",
    )
    infer_parser.set_defaults(func=infer)

    eval_parser = sub.add_parser(
        "eval",
        help="Top-1/Top-3 + 순종 sanity check",
    )
    # 기본값은 val — "모든 튜닝은 val, test는 최종 1회" (README 7 제1규칙).
    # test 평가는 마지막에 딱 한 번, --split-dir로 명시해서 실행한다.
    eval_parser.add_argument(
        "--split-dir",
        default="data/processed/breed_body/val",
        help="평가 split 디렉터리 (기본 val — test는 최종 1회만 명시 실행)",
    )
    eval_parser.add_argument(
        "--prototypes",
        default="artifacts/prototypes.npz",
        help="prototype npz",
    )
    _add_cap(eval_parser)
    eval_parser.set_defaults(func=evaluate_breed)

    ood_parser = sub.add_parser(
        "eval-ood",
        help="개 vs 고양이 max-sim AUROC",
    )
    # 기본값은 val — OOD threshold 튜닝은 val에서 하고, test는 최종 1회만 (README 7)
    ood_parser.add_argument(
        "--dog-dir",
        default="data/processed/breed_body/val",
        help="in-distribution 개 이미지 디렉터리 (기본 val)",
    )
    ood_parser.add_argument(
        "--cat-dir",
        default="data/processed/ood/val",
        help="OOD 고양이 이미지 디렉터리 (기본 val)",
    )
    ood_parser.add_argument(
        "--prototypes",
        default="artifacts/prototypes.npz",
        help="prototype npz",
    )
    _add_cap(ood_parser)
    ood_parser.set_defaults(func=evaluate_ood)

    return parser
