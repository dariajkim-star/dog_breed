"""사진 한 장이 파이프라인을 통과하는 과정을 단계별로 보여준다.

각 단계에서 데이터가 어떤 모양으로 바뀌는지 눈으로 확인하는 용도.
중간 산출물(crop 이미지)은 파일로 저장해 직접 열어볼 수 있게 한다.

사용:
    python scripts/trace_pipeline.py --image my_photos/dog.jpg
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

LINE = "=" * 66


def head(n: int, title: str, actor: str) -> None:
    print(f"\n{LINE}\n [{n}단계] {title}\n         담당: {actor}\n{LINE}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prototypes", default="artifacts/prototypes.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="artifacts/trace")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from detection import DogDetector
    from encoder import BreedEncoder
    from pipeline import load_prototypes
    from preprocessing import crop_dog, load_image
    from scoring import similarity_scores

    # ---------------------------------------------------------------- 0
    head(0, "사진 파일을 숫자 배열로 읽기", "preprocessing.load_image()")
    with Image.open(args.image) as im:
        raw_size = im.size
        orientation = im.getexif().get(274)
    img = load_image(args.image)
    print(f"  파일에 저장된 크기 : {raw_size[0]} x {raw_size[1]}")
    print(f"  EXIF 회전 태그     : {orientation}  (스마트폰이 '돌려서 보라'고 남긴 메모)")
    print(f"  회전 반영 후 배열  : {img.shape}   ← (높이, 너비, RGB 3채널)")
    print(f"  값의 범위          : {img.min()} ~ {img.max()}  (dtype={img.dtype})")
    print(f"\n  즉 사진 한 장 = 숫자 {img.size:,}개짜리 표")

    # ---------------------------------------------------------------- 1
    head(1, "사진에서 개가 '어디' 있는지 찾기", "YOLO  (또는 Faster R-CNN)")
    detector = DogDetector()
    dets = detector.detect(img)
    print(f"  이 단계는 견종을 전혀 모른다. 오직 위치만 답한다.")
    if not dets:
        print("  검출 0건 → 사진 전체를 대신 사용")
        h, w = img.shape[:2]
        bbox, conf = (0.0, 0.0, float(w), float(h)), None
    else:
        bbox, conf = dets[0]
        print(f"\n  찾은 개    : {len(dets)}마리")
        print(f"  네모 좌표  : 왼쪽 {bbox[0]:.0f}, 위 {bbox[1]:.0f}, "
              f"오른쪽 {bbox[2]:.0f}, 아래 {bbox[3]:.0f}")
        print(f"  확신도     : {conf:.3f}   (1에 가까울수록 확실)")

    # ---------------------------------------------------------------- 2
    head(2, "네모대로 잘라서 규격 맞추기", "utils.crop.standard_crop()")
    crop = crop_dog(img, bbox)
    crop_path = out_dir / "step2_crop_518px.jpg"
    Image.fromarray(crop).save(crop_path, quality=95)
    print(f"  자르는 규칙: 15% 여유 → 정사각형 → 경계 자르기 → 회색 채우기 → 518px")
    print(f"  결과 배열  : {crop.shape}")
    print(f"  저장       : {crop_path.relative_to(ROOT)}   ← 열어서 확인 가능")
    print(f"\n  왜 규격이 중요한가: 대표값을 만들 때와 지금 자르는 규칙이 다르면")
    print(f"  비교 자체가 무의미해진다. 그래서 이 함수 하나만 쓰도록 못박혀 있다.")

    # ---------------------------------------------------------------- 3
    head(3, "잘라낸 사진을 숫자 384개로 요약", "DINOv2  ← 견종 담당은 여기부터")
    encoder = BreedEncoder()
    emb = encoder.encode(crop)
    print(f"  입력  : {crop.shape}  = 숫자 {crop.size:,}개")
    print(f"  출력  : {emb.shape}        = 숫자 {emb.size}개")
    print(f"  압축비: {crop.size // emb.size:,}배로 요약됨")
    print(f"\n  앞 8개 값 : {np.array2string(emb[:8], precision=3, floatmode='fixed')}")
    print(f"  벡터 길이 : {np.linalg.norm(emb):.6f}   (항상 1로 맞춘다)")
    print(f"\n  이 384개가 '외형 요약본'이다. 귀 모양·주둥이 비율·털 질감 같은")
    print(f"  특징이 사람이 읽을 수 없는 형태로 압축돼 있다.")

    # ---------------------------------------------------------------- 4
    head(4, "126종 대표값과 하나씩 비교", "scoring.similarity_scores()")
    proto = load_prototypes(ROOT / args.prototypes)
    print(f"  미리 만들어 둔 대표값: {len(proto['prototypes'])}종")
    print(f"  대표값 하나의 모양   : {next(iter(proto['prototypes'].values())).shape}")
    print(f"\n  비교 방식: 내 강아지 벡터와 각 견종 대표 벡터의 '방향이 얼마나 같은가'")
    print(f"  (둘 다 길이가 1이라 곱해서 더하기만 하면 된다)")
    sims = similarity_scores(emb, proto)
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    print(f"\n  유사도 상위 5종:")
    for b, v in ranked[:5]:
        bar = "█" * int(v * 40)
        print(f"    {b:<24} {v:+.3f}  {bar}")
    print(f"  ...")
    print(f"    {ranked[-1][0]:<24} {ranked[-1][1]:+.3f}   ← 가장 안 닮은 견종")

    # ---------------------------------------------------------------- 5
    head(5, "답할지 말지 정하고, 퍼센트로 바꾸기", "scoring.predict()")
    max_sim = ranked[0][1]
    print(f"  1위 유사도 : {max_sim:.3f}")
    print(f"  기준선     : {args.threshold}")
    if max_sim < args.threshold:
        print(f"  판정       : 기준선 미달 → '확신 없음' 표시")
        print(f"               (순종과 뚜렷이 닮지 않음 = 믹스견 신호)")
    else:
        print(f"  판정       : 기준선 통과 → 자신 있게 답함")

    vals = np.array([v for _, v in ranked], dtype=np.float32)
    logits = vals / 0.1
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    print(f"\n  유사도를 합이 100%가 되도록 변환 (temperature 0.1):")
    top5 = 0.0
    for (b, _), pr in list(zip(ranked, probs))[:5]:
        print(f"    {b:<24} {pr * 100:5.1f}%")
        top5 += pr * 100
    print(f"    {'나머지 121종':<24} {100 - top5:5.1f}%")

    print(f"\n{LINE}")
    print(" 정리")
    print(LINE)
    print(f"  사진 {img.shape[1]}x{img.shape[0]}  =  숫자 {img.size:,}개")
    print(f"    → YOLO가 네모 하나 찾음")
    print(f"    → 518x518로 규격화  =  숫자 {crop.size:,}개")
    print(f"    → DINOv2가 384개로 요약")
    print(f"    → 126종 대표값과 비교")
    print(f"    → 상위 몇 종을 퍼센트로")
    print(LINE)


if __name__ == "__main__":
    main()
