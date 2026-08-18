"""검출 성능 평가 — 모델에 상관없이 같은 저울로 재기 위한 공용 harness.

왜 ultralytics의 `yolo val`을 쓰지 않는가:
    data/processed/detection의 라벨은 단일 클래스로 0 = dog인데,
    COCO 사전학습 YOLO는 0 = person, 16 = dog다. 그대로 val을 돌리면
    개 예측(16)은 전부 오답 처리되고, 사람을 검출하면 그게 개 정답(0)에
    매칭된다. 양방향으로 틀린 숫자가 나온다.
    라벨 2만 개를 재매핑하는 방법도 있지만, 1-stage/2-stage 비교를 하려면
    어차피 Faster R-CNN(COCO JSON 포맷)도 같은 저울로 재야 한다.
    그래서 예측을 (xyxy, conf) 목록으로 받는 공용 평가기를 둔다.

지표:
    mAP@0.5, mAP@0.5:0.95  — COCO 방식 101점 보간
    conf 0.25 기준 precision / recall / 검출 실패율

사용:
    python scripts/eval_detection.py --split test --model yolo11n.pt
    python scripts/eval_detection.py --split val  --model yolo11s.pt --limit 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

DET = ROOT / "data" / "processed" / "detection"
IOU_GRID = np.arange(0.5, 0.96, 0.05)  # COCO 0.50:0.05:0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="검출 성능 평가 (모델 공용)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--model", default="yolo11n.pt", help="ultralytics 가중치")
    p.add_argument("--limit", type=int, default=None, help="이미지 수 제한 (빠른 확인용)")
    p.add_argument("--conf", type=float, default=0.25, help="실용 지표를 잴 conf 임계값")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    """YOLO 포맷(정규화 cx cy w h) → xyxy 절대 픽셀 (N, 4)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        rows.append([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) → (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(confs: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """conf 내림차순 누적 PR 곡선 → 101점 보간 AP (COCO 방식)."""
    if n_gt == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-confs)
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(1 - is_tp[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    # precision envelope — recall이 늘어도 precision이 다시 오르지 않도록 뒤에서부터 최댓값
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    out = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(out.mean())


def main() -> None:
    from ultralytics import YOLO
    from PIL import Image

    args = parse_args()
    img_dir = DET / "images" / args.split
    lbl_dir = DET / "labels" / args.split
    if not img_dir.is_dir():
        sys.exit(f"[!] 없는 경로: {img_dir}")

    images = sorted(p for p in img_dir.iterdir() if p.is_file())
    if args.limit:
        images = images[: args.limit]
    print(f"모델: {args.model} | split: {args.split} | 이미지 {len(images):,}장")

    model = YOLO(args.model)
    # COCO 사전학습 모델은 dog가 16번. 단일 클래스로 학습한 모델이면 0번.
    dog_id = next((i for i, n in model.names.items() if n == "dog"), 0)
    print(f"이 모델의 dog 클래스 번호: {dog_id} ({model.names[dog_id]})\n")

    # mAP는 낮은 conf까지 훑어야 PR 곡선이 완성된다 (표준 관행)
    all_conf: list[float] = []
    tp_by_iou: dict[float, list[int]] = {t: [] for t in IOU_GRID}
    n_gt_total = 0
    n_no_det = 0          # conf 임계값 기준 검출 0건
    n_tp25 = n_fp25 = 0   # 실용 지표용
    t0 = time.time()

    # 한 장씩 넣는다. source에 리스트를 주면 ultralytics가 배치를 임의로 묶는데,
    # predict()의 batch 인자로는 그게 통제되지 않는다(무시됨). 그 결과
    #   - 모델마다 실제 배치가 달라져 속도 비교가 불공정해지고
    #   - VRAM 4GB에서 yolo11m이 3.24GiB 단일 할당을 시도하다 OOM으로 죽었다.
    # 한 장씩 넣으면 VRAM 최대 0.20GB로 모든 모델이 같은 조건에서 돌고,
    # 실측상 오히려 더 빠르다(yolo11m 177장: 리스트 방식 OOM -> 한 장씩 12.6초).
    for img_path in images:
        res = model.predict(source=str(img_path), conf=0.001, classes=[dog_id],
                            imgsz=args.imgsz, verbose=False)[0]
        with Image.open(img_path) as im:
            width, height = im.size
        gt = load_gt(lbl_dir / (img_path.stem + ".txt"), width, height)
        n_gt_total += len(gt)

        if res.boxes is None or len(res.boxes) == 0:
            pred, conf = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        else:
            pred = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()

        order = np.argsort(-conf)
        pred, conf = pred[order], conf[order]
        ious = iou_matrix(pred, gt)

        for thr in IOU_GRID:
            matched = np.zeros(len(gt), dtype=bool)
            for i in range(len(pred)):
                # conf 내림차순으로 아직 안 쓰인 GT 중 IoU가 가장 큰 것에 매칭
                cand = np.where(~matched & (ious[i] >= thr))[0] if len(gt) else np.array([], int)
                if len(cand):
                    best = cand[np.argmax(ious[i, cand])]
                    matched[best] = True
                    tp_by_iou[thr].append(1)
                else:
                    tp_by_iou[thr].append(0)
        all_conf.extend(conf.tolist())

        # ---- conf 임계값 기준 실용 지표 ----
        keep = conf >= args.conf
        if keep.sum() == 0:
            n_no_det += 1
        matched = np.zeros(len(gt), dtype=bool)
        for i in np.where(keep)[0]:
            cand = np.where(~matched & (ious[i] >= 0.5))[0] if len(gt) else np.array([], int)
            if len(cand):
                matched[cand[np.argmax(ious[i, cand])]] = True
                n_tp25 += 1
            else:
                n_fp25 += 1

    elapsed = time.time() - t0
    confs = np.array(all_conf, dtype=np.float32)

    aps = {}
    for thr in IOU_GRID:
        aps[thr] = average_precision(confs, np.array(tp_by_iou[thr], dtype=np.float32), n_gt_total)

    map50 = aps[IOU_GRID[0]]
    map5095 = float(np.mean(list(aps.values())))
    precision = n_tp25 / max(n_tp25 + n_fp25, 1)
    recall = n_tp25 / max(n_gt_total, 1)

    print("=" * 58)
    print(f"  GT 박스 {n_gt_total:,}개 / 이미지 {len(images):,}장")
    print("-" * 58)
    print(f"  mAP@0.5           : {map50:.4f}")
    print(f"  mAP@0.5:0.95      : {map5095:.4f}")
    print("-" * 58)
    print(f"  conf>={args.conf} 기준")
    print(f"    precision       : {precision:.4f}  ({n_tp25}/{n_tp25 + n_fp25})")
    print(f"    recall          : {recall:.4f}  ({n_tp25}/{n_gt_total})")
    print(f"    검출 0건 이미지 : {n_no_det}장 ({n_no_det / len(images) * 100:.2f}%)")
    print("-" * 58)
    print(f"  소요 {elapsed:.1f}초 ({len(images) / elapsed:.1f} img/s)")
    print("=" * 58)


if __name__ == "__main__":
    main()
