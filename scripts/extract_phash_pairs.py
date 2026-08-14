"""Task 6-3 — pHash 후보 쌍 추출 (Hamming distance ≤ threshold).

98k x 98k 전수 비교. 메모리 안전 버전:
- 청크 256행 x 전체열 XOR (uint64)
- popcount는 uint8 view + 256-entry lookup (대형 int64 중간 배열 없음)

사용:  python scripts/extract_phash_pairs.py
출력:  data/manifests/phash_pairs.parquet  (id_a, id_b, hamming, cross_dataset)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 10  # 저장은 10까지, 확정은 ≤8 / 6~10 육안 정책
CHUNK = 256

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def main():
    ph = pd.read_parquet(ROOT / "data" / "manifests" / "phash.parquet").dropna()
    ids = ph.image_id.to_numpy()
    hashes = np.array([np.uint64(int(h, 16)) for h in ph.phash_hex], dtype=np.uint64)
    n = len(hashes)
    print(f"images: {n}", flush=True)

    rows = []
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        xor = hashes[i0:i1, None] ^ hashes[None, :]          # (c, n) uint64
        d = POP8[xor.view(np.uint8)].reshape(i1 - i0, n, 8).sum(axis=2, dtype=np.uint8)
        for r in range(i1 - i0):
            gi = i0 + r
            js = np.nonzero(d[r, gi + 1:] <= THRESHOLD)[0] + gi + 1
            rows.extend((ids[gi], ids[j], int(d[r, j])) for j in js)
        if (i0 // CHUNK) % 20 == 0:
            print(f"{i1}/{n}  pairs so far: {len(rows)}", flush=True)

    out = pd.DataFrame(rows, columns=["id_a", "id_b", "hamming"])
    out["src_a"] = out.id_a.str.split("/").str[0]
    out["src_b"] = out.id_b.str.split("/").str[0]
    out["cross_dataset"] = out.src_a != out.src_b
    out.to_parquet(ROOT / "data" / "manifests" / "phash_pairs.parquet", index=False)

    print("\n=== summary ===", flush=True)
    print("total pairs (<=10):", len(out))
    print("  <=5 (사실상 확정):", int((out.hamming <= 5).sum()))
    print("  6~8 (확정 후보):  ", int(((out.hamming >= 6) & (out.hamming <= 8)).sum()))
    print("  9~10 (육안 필요): ", int((out.hamming >= 9).sum()))
    print("cross-dataset pairs:", int(out.cross_dataset.sum()))
    if out.cross_dataset.any():
        print(out[out.cross_dataset].groupby(["src_a", "src_b"]).size())


if __name__ == "__main__":
    sys.exit(main())
