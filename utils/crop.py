"""standard_crop() — 프로젝트 유일의 crop 규칙 (docs/PREPROCESSING.md §2).

이 함수 외의 crop 구현 금지. prototype 구축·inference·평가 전부 이 함수를 쓴다.
규칙: 각 변 expand → 경계 clamp → 긴 변 square(창 확장) → 부족분 padding → resize
"""
from PIL import Image

PAD_COLOR = (114, 114, 114)  # YOLO 관행의 회색 — 어느 쪽이든 한 가지로 고정이 핵심


def standard_crop(im: Image.Image, bbox, expand: float = 0.15, out_size: int = 518) -> Image.Image:
    """bbox: (x1, y1, x2, y2) absolute pixel. 반환: out_size x out_size RGB."""
    W, H = im.size
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2 - x1, y2 - y1

    # ① expand
    x1 -= bw * expand; x2 += bw * expand
    y1 -= bh * expand; y2 += bh * expand

    # ③ 긴 변 기준 square (창 확장, 중심 고정)
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx - side / 2, cx + side / 2
    y1, y2 = cy - side / 2, cy + side / 2

    # ② clamp
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(W, int(round(x2))), min(H, int(round(y2)))
    crop = im.convert("RGB").crop((ix1, iy1, ix2, iy2))

    # ④ 부족분 padding으로 정사각 유지 (왜곡 0)
    cw, ch = crop.size
    if cw != ch:
        side_px = max(cw, ch)
        canvas = Image.new("RGB", (side_px, side_px), PAD_COLOR)
        canvas.paste(crop, ((side_px - cw) // 2, (side_px - ch) // 2))
        crop = canvas

    return crop.resize((out_size, out_size), Image.BILINEAR)
