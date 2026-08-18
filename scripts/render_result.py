"""추론 결과를 그림 한 장으로 만든다 — 왼쪽 사진(bbox 표시) + 오른쪽 비율 막대.

발표 자료·정성평가용. 화면 출력만 하고 사라지던 결과를 파일로 남긴다.

사용:
    python scripts/render_result.py --image my_photos/dog.jpg
    python scripts/render_result.py --dir my_photos          # 폴더 통째로
출력:
    artifacts/results/<파일명>_result.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 색 (RGB)
BG = (255, 255, 255)
INK = (26, 30, 36)
MUTED = (122, 132, 145)
RULE = (222, 226, 232)
BAR = (47, 125, 95)
BAR_BG = (233, 238, 235)
BOX = (232, 93, 46)
WARN = (168, 72, 29)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"

PHOTO_W = 620          # 왼쪽 사진 영역 너비
PANEL_W = 470          # 오른쪽 패널 너비
PAD = 28


def load_fonts() -> dict:
    """윈도우 기본 한글 폰트. 없으면 PIL 기본 폰트로 떨어진다."""
    try:
        return {
            "title": ImageFont.truetype(FONT_BOLD, 27),
            "breed": ImageFont.truetype(FONT_PATH, 20),
            "pct": ImageFont.truetype(FONT_BOLD, 20),
            "small": ImageFont.truetype(FONT_PATH, 16),
            "tiny": ImageFont.truetype(FONT_PATH, 14),
        }
    except OSError:
        d = ImageFont.load_default()
        return {k: d for k in ("title", "breed", "pct", "small", "tiny")}


def render(image_path: Path, prototypes: Path, threshold: float,
           top_k: int, out_dir: Path) -> Path:
    from detection import DogDetector
    from encoder import BreedEncoder
    from pipeline import load_prototypes
    from preprocessing import crop_dog, load_image
    from scoring import predict

    fonts = load_fonts()
    img = load_image(image_path)          # EXIF 회전까지 반영된 배열
    src = Image.fromarray(img)

    # ---- 추론 ----
    dets = DogDetector().detect(img)
    if dets:
        bbox, conf = dets[0]
        fallback = False
    else:
        h, w = img.shape[:2]
        bbox, conf, fallback = (0.0, 0.0, float(w), float(h)), None, True

    crop = crop_dog(img, bbox)
    emb = BreedEncoder().encode(crop)
    result = predict(emb, load_prototypes(prototypes),
                     threshold=threshold, top_k=top_k)

    # ---- 왼쪽: 사진에 bbox 그리기 ----
    scale = PHOTO_W / src.width
    photo = src.resize((PHOTO_W, max(1, int(src.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(photo)
    if not fallback:
        x1, y1, x2, y2 = (v * scale for v in bbox)
        # 두껍게 그려야 축소된 사진에서도 보인다
        draw.rectangle([x1, y1, x2, y2], outline=BOX, width=5)
        label = f"dog {conf:.2f}"
        tw = draw.textlength(label, font=fonts["small"])
        ly = max(0, y1 - 26)
        draw.rectangle([x1, ly, x1 + tw + 14, ly + 26], fill=BOX)
        draw.text((x1 + 7, ly + 3), label, font=fonts["small"], fill=(255, 255, 255))

    canvas_h = max(photo.height + PAD * 2, 520)
    canvas = Image.new("RGB", (PHOTO_W + PANEL_W + PAD * 3, canvas_h), BG)
    canvas.paste(photo, (PAD, PAD))

    # ---- 오른쪽: 결과 패널 ----
    d = ImageDraw.Draw(canvas)
    x = PAD * 2 + PHOTO_W
    y = PAD

    d.text((x, y), "Phenotype Similarity", font=fonts["title"], fill=INK)
    y += 34
    d.text((x, y), "DNA 혈통 비율이 아닙니다", font=fonts["tiny"], fill=MUTED)
    y += 30
    d.line([(x, y), (x + PANEL_W - PAD, y)], fill=RULE, width=1)
    y += 18

    d.text((x, y), f"최고 유사도 {result['max_sim']:.3f}   (기준선 {threshold})",
           font=fonts["small"], fill=MUTED)
    y += 26

    if fallback:
        d.text((x, y), "* 개를 못 찾아 사진 전체로 추론", font=fonts["tiny"], fill=WARN)
        y += 22
    if result["unknown"]:
        d.text((x, y), "확신 낮음 — 믹스견 가능성", font=fonts["small"], fill=WARN)
        y += 26

    y += 8
    top = result["topk"]
    widest = max((pct for _, pct in top), default=1.0)
    bar_w = PANEL_W - PAD
    for breed, pct in top:
        d.text((x, y), breed.replace("_", " "), font=fonts["breed"], fill=INK)
        pct_txt = f"{pct:.1f}%"
        tw = d.textlength(pct_txt, font=fonts["pct"])
        d.text((x + bar_w - tw, y), pct_txt, font=fonts["pct"], fill=BAR)
        y += 27
        d.rectangle([x, y, x + bar_w, y + 9], fill=BAR_BG)
        # 1위를 꽉 차게 그려 상대 비교가 눈에 들어오게 한다
        filled = int(bar_w * (pct / widest)) if widest else 0
        if filled > 0:
            d.rectangle([x, y, x + filled, y + 9], fill=BAR)
        y += 26

    other = 100.0 - sum(pct for _, pct in top)
    if other > 0.05:
        d.text((x, y), f"나머지 {other:.1f}%", font=fonts["small"], fill=MUTED)

    d.text((x, canvas_h - PAD - 16), image_path.name, font=fonts["tiny"], fill=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{image_path.stem}_result.png"
    canvas.save(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="추론 결과를 그림으로 저장")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="사진 한 장")
    g.add_argument("--dir", help="폴더 전체")
    p.add_argument("--prototypes", default="artifacts/prototypes.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=5)
    # 기본 출력은 artifacts/ 아래로 — data/reports는 git이 추적하는 폴더라
    # 개인 사진이 들어간 결과 이미지가 public 저장소에 올라간다.
    p.add_argument("--out", default="artifacts/results")
    args = p.parse_args()

    if args.image:
        targets = [Path(args.image)]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        targets = sorted(q for q in Path(args.dir).iterdir()
                         if q.is_file() and q.suffix.lower() in exts)
    if not targets:
        sys.exit("[!] 처리할 이미지가 없습니다.")

    out_dir = ROOT / args.out
    for q in targets:
        out = render(q, ROOT / args.prototypes, args.threshold, args.top_k, out_dir)
        print(f"  저장: {out.relative_to(ROOT)}")
    print(f"\n{len(targets)}장 완료 -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
