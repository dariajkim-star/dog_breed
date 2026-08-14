"""Taxonomy FREEZE — 25종 확정, manifest에 canonical_breed / breed_id 부여.

결정 사항 (8/14, 권장안 채택):
  - teddy → poodle 병합 (푸들 미용 스타일)
  - chinese_rural_dog → DROP (순종 아님)
  - cardigan + pembroke → corgi 병합
  - oxford english_cocker_spaniel → cocker_spaniel에 병합하지 않음 (다른 견종) → 25종 밖

출력: data/manifests/taxonomy.csv (FREEZE), master_manifest.parquet 갱신
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MVP25 = [
    "basset", "beagle", "bichon_frise", "border_collie", "chihuahua",
    "cocker_spaniel", "corgi", "french_bulldog", "german_shepherd",
    "golden_retriever", "labrador_retriever", "malamute", "maltese_dog",
    "miniature_pinscher", "miniature_schnauzer", "papillon", "pomeranian",
    "poodle", "pug", "samoyed", "shiba_inu", "shih_tzu",
    "siberian_husky", "staffordshire_bullterrier", "yorkshire_terrier",
]
BREED_ID = {b: i for i, b in enumerate(MVP25)}  # 알파벳순 0~24

FIX = {
    "toy_poodle": "poodle", "miniature_poodle": "poodle", "standard_poodle": "poodle",
    "teddy": "poodle",
    "shiba_dog": "shiba_inu",
    "cardigan": "corgi", "pembroke": "corgi",
    "staffordshire_bull_terrier": "staffordshire_bullterrier",
    "basset_hound": "basset",
    "shih-tzu": "shih_tzu",
    "maltese": "maltese_dog",
    "chinese_rural_dog": "__DROP__",
    "english_cocker_spaniel": "__NOT_MVP__",  # 미국 cocker와 다른 견종 - 병합 금지
}


def canon(label: str) -> str:
    b = str(label).lower().replace("-", "_").replace(" ", "_")
    b = FIX.get(b, b)
    return b


def main():
    mpath = ROOT / "data" / "manifests" / "master_manifest.parquet"
    df = pd.read_parquet(mpath)

    is_cat = df.original_breed.str.startswith("CAT_")
    df["canonical_breed"] = None
    df.loc[~is_cat, "canonical_breed"] = df.loc[~is_cat, "original_breed"].map(canon)
    df.loc[is_cat, "canonical_breed"] = "__CAT_OOD__"
    df["breed_id"] = df.canonical_breed.map(BREED_ID).astype("Int64")
    df["in_mvp25"] = df.breed_id.notna() & df.usable

    df.to_parquet(mpath, index=False)

    # taxonomy.csv (freeze)
    rows = []
    for (src, orig), grp in df[~is_cat].groupby(["source", "original_breed"]):
        c = grp.canonical_breed.iloc[0]
        action = ("drop" if c == "__DROP__" else
                  "not_mvp" if c not in BREED_ID and c != "__DROP__" else
                  "mvp25")
        rows.append(dict(source=src, original_label=orig, canonical_label=c,
                         breed_id=BREED_ID.get(c, ""), action=action, n_images=len(grp)))
    tax = pd.DataFrame(rows).sort_values(["action", "canonical_label", "source"])
    tax.to_csv(ROOT / "data" / "manifests" / "taxonomy.csv", index=False, encoding="utf-8-sig")

    mvp = df[df.in_mvp25]
    print("MVP25 usable images:", len(mvp))
    print(mvp.groupby("canonical_breed").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
