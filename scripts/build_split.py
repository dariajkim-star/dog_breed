"""Dedup 그룹핑 + Stratified Group Split (70/15/15) — SPLIT FREEZE.

- 그룹: pHash Hamming <= 10 전체를 보수적으로 연결 (놓침 >> 과다묶음 원칙,
  embedding dedup은 시간상 생략 - 개선 트랙)
- split 단위 = dedup group (그룹 통째로 한 split에)
- 품종별로 group을 크기 내림차순 greedy 배정 → 70/15/15 근사
- 고양이(__CAT_OOD__): 학습 안 하므로 val/test 50:50
- 출력: manifest에 dedup_group/split 컬럼, train/val/test.parquet, split_report
"""
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


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
    random.seed(42)
    mpath = ROOT / "data" / "manifests" / "master_manifest.parquet"
    df = pd.read_parquet(mpath)

    pairs = pd.read_parquet(ROOT / "data" / "manifests" / "phash_pairs.parquet")
    dsu = DSU()
    for r in pairs.itertuples():          # <=10 전체 (보수적)
        dsu.union(r.id_a, r.id_b)

    df["dedup_group"] = [f"DG_{hash(dsu.find(i)) & 0xFFFFFFFF:08x}" if i in dsu.p
                         else f"SG_{k}" for k, i in enumerate(df.image_id)]

    # ---- 25종 split (group 단위, 품종별 greedy) ----
    target = df[df.in_mvp25].copy()
    df["split"] = None
    group_breed = target.groupby("dedup_group").agg(
        n=("image_id", "count"),
        breed=("canonical_breed", lambda s: s.mode().iloc[0]),
    ).reset_index()

    assign = {}
    for breed, g in group_breed.groupby("breed"):
        g = g.sample(frac=1, random_state=42).sort_values("n", ascending=False)
        total = g.n.sum()
        filled = {k: 0 for k in RATIOS}
        for row in g.itertuples():
            # 현재 비율 대비 가장 부족한 split에 배정
            deficit = {k: RATIOS[k] - filled[k] / total for k in RATIOS}
            best = max(deficit, key=deficit.get)
            assign[row.dedup_group] = best
            filled[best] += row.n

    mvp_mask = df.in_mvp25
    df.loc[mvp_mask, "split"] = df.loc[mvp_mask, "dedup_group"].map(assign)

    # ---- 고양이 OOD: val/test 반반 ----
    cat_mask = df.canonical_breed == "__CAT_OOD__"
    cats = df[cat_mask].image_id.tolist()
    random.shuffle(cats)
    half = len(cats) // 2
    df.loc[df.image_id.isin(cats[:half]), "split"] = "val"
    df.loc[df.image_id.isin(cats[half:]), "split"] = "test"

    df.to_parquet(mpath, index=False)
    for s in ("train", "val", "test"):
        part = df[(df.split == s) & df.in_mvp25]
        part.to_parquet(ROOT / "data" / "manifests" / f"{s}.parquet", index=False)

    # ---- 리포트 ----
    tab = df[df.in_mvp25].groupby(["canonical_breed", "split"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab["train%"] = (tab.get("train", 0) / tab.total * 100).round(1)
    lines = ["# Split Report (FREEZE)", "",
             f"- 그룹핑: pHash <=10 전체 연결 (embedding dedup은 개선 트랙)",
             f"- 25종 이미지: {int(df.in_mvp25.sum()):,} / 고양이 OOD: {int(cat_mask.sum()):,} (val/test 반반)",
             "", tab.to_markdown()]
    (ROOT / "data" / "reports" / "split_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string())
    print("\nsplit sizes:", df[df.in_mvp25].split.value_counts().to_dict())


if __name__ == "__main__":
    main()
