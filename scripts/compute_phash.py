"""Task 6-2 — pHash 계산 (dedup 2단계 준비).

master_manifest의 usable 이미지 전체에 대해 64-bit pHash를 계산해 저장한다.
후보 쌍 추출(BK-tree/전수 XOR)은 다음 단계 스크립트에서 수행.

사용:  python scripts/compute_phash.py
출력:  data/manifests/phash.parquet  (image_id, phash_hex)
"""
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "phash.parquet"


def main():
    df = pd.read_parquet(ROOT / "data" / "manifests" / "master_manifest.parquet")
    df = df[df.usable | (df.exclusion_reason == "cat_ood_only")]  # 고양이도 OOD셋이므로 dedup 대상
    print(f"target images: {len(df)}")

    ids, hashes = [], []
    for image_id, rel in tqdm(zip(df.image_id, df.image_path), total=len(df)):
        try:
            with Image.open(ROOT / rel) as im:
                h = imagehash.phash(im.convert("RGB"))  # 8x8 DCT, 64-bit
            ids.append(image_id)
            hashes.append(str(h))
        except Exception as e:
            ids.append(image_id)
            hashes.append(None)  # corrupt 등 — manifest의 usable과 교차 확인용

    pd.DataFrame({"image_id": ids, "phash_hex": hashes}).to_parquet(OUT, index=False)
    n_fail = sum(h is None for h in hashes)
    print(f"done: {len(ids)} hashed, {n_fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
