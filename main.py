"""FindDogBreed CLI — 얇은 엔트리포인트 (인자 파싱 + 출력만).

실제 로직은 역할별 모듈에 있다:
    preprocessing.py : 이미지 로드 / crop / split 순회
    model.py         : YOLO detector / DINOv2 encoder / prototype / predict
    pipeline.py      : 단계 실행 (embed, prototype, infer)
    evaluate.py      : 평가지표 (Top-1/3, OOD AUROC)

서브커맨드는 파이프라인의 한 단계만 수행한다 (README 아키텍처 참조).
무거운 torch 계열 import는 각 함수 내부에서 lazy import —
`python main.py --help`가 빠르게 뜨도록 유지.
"""

from __future__ import annotations

import argparse
import sys

# Unknown 출력 문구 — README 4. Inference 참조
UNKNOWN_MESSAGE = "현재 지원하는 품종만으로 설명하기 어렵습니다"


def cmd_detect(args: argparse.Namespace) -> None:
    """detect: 사진 한 장에서 강아지 bbox/confidence 출력 (Stage 1)."""
    from preprocessing import load_image
    from model import DogDetector

    img = load_image(args.image)
    detections = DogDetector(weights=args.weights, conf=args.conf).detect(img)

    if not detections:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return
    print(f"탐지 결과: {len(detections)}건")
    for i, (bbox, conf) in enumerate(detections, start=1):
        x1, y1, x2, y2 = bbox
        print(f"  [{i}] bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  conf={conf:.3f}")


def cmd_embed(args: argparse.Namespace) -> None:
    """embed: crop 완료 split → embedding npz 저장 (Stage 2)."""
    from pipeline import embed_split

    print(f"split 순회 중: {args.split_dir}")
    n, shape = embed_split(args.split_dir, args.out, cap=args.cap)
    print(f"저장 완료: {args.out}  ({n}장, embeddings={shape})")


def cmd_prototype(args: argparse.Namespace) -> None:
    """prototype: embedding npz → 견종 prototype npz 생성 (Stage 3)."""
    from pipeline import build_and_save_prototypes

    breeds = build_and_save_prototypes(args.embeddings, args.out, cap=args.cap)
    print(f"저장 완료: {args.out}  ({len(breeds)}종)")


def cmd_infer(args: argparse.Namespace) -> None:
    """infer: 단일 이미지 전체 경로 실행 후 결과 pretty-print (README 4)."""
    from pipeline import infer_image

    result = infer_image(args.image, args.prototypes, threshold=args.threshold,
                         top_k=args.top_k, temperature=args.temperature)
    if result is None:
        print("강아지를 찾지 못했습니다. (detection 0건)")
        return

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
    """eval: Top-1/Top-3 accuracy + 순종 sanity check (README 7)."""
    from evaluate import evaluate_split

    print(f"평가 시작: {args.split_dir}")
    r = evaluate_split(args.split_dir, args.prototypes, cap=args.cap)
    print("=== 평가 결과 (README 7. 평가지표) ===")
    print(f"Top-1 accuracy            : {r['top1']:.4f}  ({r['top1_hit']}/{r['n']})")
    print(f"Top-3 accuracy            : {r['top3']:.4f}  ({r['top3_hit']}/{r['n']})")
    print(f"순종 sanity check (Top-1) : {r['purebred_sanity']:.4f}")


def cmd_eval_ood(args: argparse.Namespace) -> None:
    """eval-ood: 개 vs 고양이 max-sim AUROC (README 7, OOD)."""
    from evaluate import evaluate_ood

    print(f"개(in-dist): {args.dog_dir} / 고양이(OOD): {args.cat_dir}")
    r = evaluate_ood(args.dog_dir, args.cat_dir, args.prototypes, cap=args.cap)
    print("=== OOD 평가 결과 (README 7. 평가지표) ===")
    print(f"개 {r['n_dog']}장 / 고양이 {r['n_cat']}장")
    print(f"max-similarity AUROC : {r['auroc']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FindDogBreed 파이프라인 — 단계별 실행 (한 번에 한 단계만)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cap(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cap", type=int, default=None,
                       help="클래스당 최대 장수 (CPU 테스트 런 용도)")

    p = sub.add_parser("detect", help="사진에서 강아지 bbox 탐지 (Stage 1)")
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO 가중치 (기본 yolo11n.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence 임계값")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("embed", help="crop 완료 split → embedding npz 저장 (Stage 2)")
    p.add_argument("--split-dir", default="data/processed/breed_body/train",
                   help="이미 crop된 split 디렉토리")
    p.add_argument("--out", default="artifacts/embeddings_train.npz", help="출력 npz 경로")
    add_cap(p)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("prototype", help="embedding npz → 견종 prototype 생성 (Stage 3)")
    p.add_argument("--embeddings", default="artifacts/embeddings_train.npz",
                   help="embed 단계 출력 npz")
    p.add_argument("--out", default="artifacts/prototypes.npz", help="출력 npz 경로")
    p.add_argument("--cap", type=int, default=50, help="클래스당 이미지 캡 (README 3)")
    p.set_defaults(func=cmd_prototype)

    p = sub.add_parser("infer", help="단일 이미지 추론 (detect→crop→encode→predict)")
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Unknown 분기 threshold (val에서 결정)")
    p.add_argument("--top-k", type=int, default=3, help="상위 K개 견종 출력")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="calibration temperature (README 5)")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("eval", help="Top-1/Top-3 + 순종 sanity check (README 7)")
    p.add_argument("--split-dir", default="data/processed/breed_body/test",
                   help="평가 split 디렉토리 (test는 최종 1회만!)")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    add_cap(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval-ood", help="개 vs 고양이 max-sim AUROC (README 7, OOD)")
    p.add_argument("--dog-dir", default="data/processed/breed_body/test",
                   help="in-distribution 개 이미지 디렉토리")
    p.add_argument("--cat-dir", default="data/processed/ood/test",
                   help="OOD 고양이 이미지 디렉토리")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz", help="prototype npz")
    add_cap(p)
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
