"""Unknown threshold 튜닝 — 개/고양이 max-similarity 분포에서 운영점을 고른다.

eval-ood는 AUROC 한 숫자만 준다. AUROC는 "분리가 얼마나 잘 되나"는 알려주지만
"threshold를 얼마로 둘까"는 답하지 않는다. 이 스크립트는 두 분포를 실제로 뽑아
threshold를 바꿔가며 다음 두 값을 함께 보여준다:

    개 통과율   (recall)      — 진짜 개를 Unknown으로 잘못 거절하지 않는 비율
    고양이 통과율 (false accept) — 개가 아닌 것에 억지로 견종을 답하는 비율

README 7 원칙대로 튜닝은 val에서만 한다. test는 최종 1회.

사용:
    python scripts/tune_threshold.py --cap 30 --cat-cap 1200 --batch-size 8
출력:
    data/reports/threshold_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unknown threshold 튜닝 (val 전용)")
    p.add_argument("--dog-dir", default="data/processed/breed_body/val")
    p.add_argument("--cat-dir", default="data/processed/ood/val")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz")
    p.add_argument("--cap", type=int, default=30, help="견종당 개 이미지 수")
    p.add_argument("--cat-cap", type=int, default=None, help="고양이 이미지 수 (기본: 전부)")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def describe(name: str, v: np.ndarray) -> str:
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return (f"{name:8s} n={len(v):5,}  min={v.min():.3f}  "
            f"p1={q[0]:.3f} p5={q[1]:.3f} p25={q[2]:.3f} p50={q[3]:.3f} "
            f"p75={q[4]:.3f} p95={q[5]:.3f} p99={q[6]:.3f}  max={v.max():.3f}")


def main() -> None:
    from evaluate import auroc_by_rank, max_sims_for_dir
    from encoder import BreedEncoder
    from pipeline import load_prototypes

    args = parse_args()
    proto = load_prototypes(ROOT / args.prototypes)

    # encoder를 한 번만 만들어 두 디렉터리에 재사용 (DINOv2 재로딩 방지)
    encoder = BreedEncoder()

    print(f"개  : {args.dog_dir} (견종당 {args.cap}장)")
    dog = np.array(max_sims_for_dir(ROOT / args.dog_dir, proto, cap=args.cap,
                                    encoder=encoder, batch_size=args.batch_size))
    print(f"고양이: {args.cat_dir} (cap={args.cat_cap})")
    cat = np.array(max_sims_for_dir(ROOT / args.cat_dir, proto, cap=args.cat_cap,
                                    encoder=encoder, batch_size=args.batch_size))

    auroc = auroc_by_rank(list(dog), list(cat))

    lines: list[str] = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# Unknown Threshold 튜닝 리포트 (val)")
    out()
    out(f"- prototype: {args.prototypes} ({len(proto['prototypes'])}종)")
    out(f"- 개 {len(dog):,}장 / 고양이 {len(cat):,}장")
    out(f"- **max-similarity AUROC: {auroc:.4f}**")
    out()
    out("## max-similarity 분포")
    out()
    out("```")
    out(describe("개", dog))
    out(describe("고양이", cat))
    out("```")
    out()

    # threshold 후보: 두 분포를 아우르는 구간을 훑는다
    lo = min(dog.min(), cat.min())
    hi = max(dog.max(), cat.max())
    grid = np.round(np.linspace(lo, hi, 25), 3)

    out("## threshold 스윕")
    out()
    out("| threshold | 개 통과율 (높을수록 좋음) | 고양이 통과율 (낮을수록 좋음) |")
    out("|---:|---:|---:|")
    for t in grid:
        out(f"| {t:.3f} | {(dog >= t).mean()*100:6.2f}% | {(cat >= t).mean()*100:6.2f}% |")
    out()

    # 운영점 추천 — 목표를 먼저 정하고 그에 맞는 threshold를 역산한다
    out("## 운영점 후보")
    out()
    out("| 목표 | threshold | 개 통과율 | 고양이 통과율 |")
    out("|:--|---:|---:|---:|")
    for label, t in [
        ("개 99% 통과 (거절 최소)", float(np.percentile(dog, 1))),
        ("개 95% 통과 (권장)", float(np.percentile(dog, 5))),
        ("개 90% 통과 (거절 강화)", float(np.percentile(dog, 10))),
        ("고양이 5% 통과", float(np.percentile(cat, 95))),
        ("고양이 1% 통과", float(np.percentile(cat, 99))),
        ("고양이 0% 통과 (최댓값)", float(cat.max())),
    ]:
        out(f"| {label} | {t:.3f} | {(dog >= t).mean()*100:6.2f}% | {(cat >= t).mean()*100:6.2f}% |")
    out()
    out(f"현재 기본값 0.5 → 개 {(dog >= 0.5).mean()*100:.2f}% 통과 / "
        f"고양이 {(cat >= 0.5).mean()*100:.2f}% 통과")

    report = ROOT / "data" / "reports" / "threshold_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
