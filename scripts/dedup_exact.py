"""확정 중복(exact duplicate) 제거 — keep-one 정책.

정의: MD5 동일 OR pHash Hamming ≤ 5 (정본 문서의 "자동 확정" 구간)
정책 (8/14 사용자 결정):
  - connected component로 중복 그룹 구성
  - 그룹당 1장만 유지. 유지 우선순위 = 전체 데이터가 적은 소스부터
    oxford(7,390) > stanford(20,580) > tsinghua(70,432)  ← 적은 쪽에 남김
  - 원본(raw)은 삭제하지 않음. manifest에 usable=False + exclusion_reason='exact_duplicate'
    + dup_keep_id(유지된 대표 image_id) 기록
  - data/labeled/ 에서는 탈락본의 hardlink + label txt 제거 (한쪽만 남김)
  - Hamming 6~10 및 embedding 단계 후보는 여기서 처리하지 않음 → dedup_group으로 남겨 group split에서 처리

사용:  python scripts/dedup_exact.py
출력:  master_manifest.parquet 갱신(v4), data/reports/exact_dedup_report.md
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEEP_PRIORITY = {"oxford": 0, "stanford": 1, "tsinghua": 2}  # 낮을수록 유지


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    mpath = ROOT / "data" / "manifests" / "master_manifest.parquet"
    df = pd.read_parquet(mpath)
    breed_src = df.set_index("image_id")

    dsu = DSU()

    # 1) MD5 동일 그룹
    md5_groups = df[df.usable].groupby("md5").image_id.apply(list)
    n_md5 = 0
    for ids in md5_groups:
        if len(ids) > 1:
            n_md5 += 1
            for other in ids[1:]:
                dsu.union(ids[0], other)

    # 2) pHash Hamming <= 5
    pairs = pd.read_parquet(ROOT / "data" / "manifests" / "phash_pairs.parquet")
    strict = pairs[pairs.hamming <= 5]
    for r in strict.itertuples():
        dsu.union(r.id_a, r.id_b)

    # 3) 그룹별 keeper 선정
    groups = defaultdict(list)
    for x in list(dsu.p):
        groups[dsu.find(x)].append(x)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    def keep_key(image_id):
        src = image_id.split("/")[0]
        return (KEEP_PRIORITY.get(src, 9), image_id)

    drop, keep_of = [], {}
    for members in groups.values():
        members.sort(key=keep_key)
        keeper = members[0]
        for m in members[1:]:
            drop.append(m)
            keep_of[m] = keeper

    # 4) manifest 갱신
    df["dup_keep_id"] = None
    mask = df.image_id.isin(drop)
    df.loc[mask & df.usable, "exclusion_reason"] = "exact_duplicate"
    df.loc[mask, "usable"] = False
    df.loc[mask, "dup_keep_id"] = df.loc[mask, "image_id"].map(keep_of)
    df.to_parquet(mpath, index=False)

    # 5) data/labeled 에서 탈락본 제거 (hardlink + label)
    removed_files = 0
    for image_id in drop:
        parts = image_id.split("/")
        if len(parts) == 3:                        # tsinghua/stanford: src/breed_dir/fname
            src, breed_dir, fname = parts
        else:                                      # oxford: src/fname → 견종은 파일명에서
            src, fname = parts
            breed_dir = fname.rsplit("_", 1)[0]
        img = ROOT / "data" / "labeled" / src / "images" / breed_dir / fname
        lbl = ROOT / "data" / "labeled" / src / "labels" / breed_dir / (Path(fname).stem + ".txt")
        for p in (img, lbl):
            if p.exists():
                p.unlink()
                removed_files += 1

    # 6) 리포트
    drop_by_src = pd.Series([d.split("/")[0] for d in drop]).value_counts()
    lines = [
        "# Exact Dedup Report (keep-one)", "",
        f"- 기준: MD5 동일 OR pHash Hamming <= 5",
        f"- 정책: 그룹당 1장 유지, 유지 우선순위 oxford > stanford > tsinghua (데이터 적은 소스)",
        f"- 중복 그룹 수: {len(groups):,}",
        f"- 제외된 이미지: {len(drop):,}", "",
        "## 소스별 제외", "", drop_by_src.to_markdown(), "",
        f"- data/labeled 에서 제거된 파일(이미지+라벨): {removed_files:,}",
        f"- raw 원본은 보존, manifest에 exclusion_reason='exact_duplicate' + dup_keep_id 기록",
    ]
    (ROOT / "data" / "reports" / "exact_dedup_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"groups: {len(groups):,}  dropped: {len(drop):,}")
    print(drop_by_src)
    print("usable now:", int(df.usable.sum()))


if __name__ == "__main__":
    main()
