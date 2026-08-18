"""Taxonomy FREEZE v2 — 126종 확정, manifest에 canonical_breed / breed_id 부여.

v1(25종) 대비 변경 (8/18):
  - MVP25 -> BREEDS 126종으로 확대 (Stanford 120 + Tsinghua 130 + Oxford 25 어휘 통합)
  - 소스별 철자 변형 6쌍 병합 (아래 FIX v2 신규 항목 참조)
  - english_cocker_spaniel 복원 — 미국 cocker와 다른 견종이므로 별도 클래스
  - 개가 아닌 3종(dhole, dingo, african_hunting_dog)은 __NOT_DOG__로 분리 -> hard OOD 후보

v1에서 그대로 유지되는 결정:
  - teddy -> poodle 병합 (푸들 미용 스타일)
  - chinese_rural_dog -> DROP (순종 아님)
  - cardigan + pembroke -> corgi 병합

출력: data/manifests/taxonomy.csv (FREEZE), master_manifest.parquet 갱신
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 126종 — 알파벳순. breed_id가 이 순서대로 0~125로 매겨지므로 순서를 지킬 것.
BREEDS = [
    "affenpinscher", "afghan_hound", "airedale", "american_bulldog",
    "american_pit_bull_terrier", "american_staffordshire_terrier", "appenzeller", "australian_shepherd",
    "australian_terrier", "basenji", "basset", "beagle",
    "bedlington_terrier", "bernese_mountain_dog", "bichon_frise", "black_and_tan_coonhound",
    "black_sable", "blenheim_spaniel", "bloodhound", "bluetick",
    "border_collie", "border_terrier", "borzoi", "boston_bull",
    "bouvier_des_flandres", "boxer", "brabancon_griffo", "briard",
    "brittany_spaniel", "bull_mastiff", "cairn", "cane_carso",
    "chesapeake_bay_retriever", "chihuahua", "chinese_crested_dog", "chow",
    "clumber", "cocker_spaniel", "collie", "corgi",
    "curly_coated_retriever", "dandie_dinmont", "doberman", "english_cocker_spaniel",
    "english_foxhound", "english_setter", "english_springer", "entlebucher",
    "eskimo_dog", "fila_braziliero", "flat_coated_retriever", "french_bulldog",
    "german_shepherd", "german_short_haired_pointer", "giant_schnauzer", "golden_retriever",
    "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog",
    "groenendael", "havanese", "ibizan_hound", "irish_setter",
    "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "italian_greyhound",
    "japanese_spaniel", "japanese_spitzes", "keeshond", "kelpie",
    "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever",
    "lakeland_terrier", "leonberg", "lhasa", "malamute",
    "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher",
    "miniature_schnauzer", "newfoundland", "norfolk_terrier", "norwegian_elkhound",
    "norwich_terrier", "old_english_sheepdog", "otterhound", "papillon",
    "pekinese", "pomeranian", "poodle", "pug",
    "redbone", "rhodesian_ridgeback", "rottweiler", "saint_bernard",
    "saluki", "samoyed", "schipperke", "scotch_terrier",
    "scottish_deerhound", "sealyham_terrier", "shetland_sheepdog", "shiba_inu",
    "shih_tzu", "siberian_husky", "silky_terrier", "soft_coated_wheaten_terrier",
    "staffordshire_bullterrier", "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff",
    "tibetan_terrier", "toy_terrier", "vizsla", "walker_hound",
    "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet",
    "wire_haired_fox_terrier", "yorkshire_terrier",
]
assert BREEDS == sorted(BREEDS), "BREEDS는 알파벳순이어야 합니다"
assert len(BREEDS) == len(set(BREEDS)), "BREEDS에 중복 항목이 있습니다"

BREED_ID = {b: i for i, b in enumerate(BREEDS)}  # 알파벳순 0~125

# 개가 아님 — 견종 클래스에서 빼고 hard OOD 세트로 재활용한다 (SESSION_HANDOVER 참조).
NOT_DOG = {"dhole", "dingo", "african_hunting_dog"}

FIX = {
    # ---- v1에서 이어받은 병합 ----
    "toy_poodle": "poodle", "miniature_poodle": "poodle", "standard_poodle": "poodle",
    "teddy": "poodle",
    "shiba_dog": "shiba_inu",
    "cardigan": "corgi", "pembroke": "corgi",
    "staffordshire_bull_terrier": "staffordshire_bullterrier",
    "basset_hound": "basset",
    "shih-tzu": "shih_tzu",
    "maltese": "maltese_dog",
    "chinese_rural_dog": "__DROP__",

    # ---- v2 신규: 소스별 철자 변형 통일 ----
    # 같은 견종이 두 클래스로 갈라지면 prototype이 서로를 잡아먹으므로 반드시 병합.
    "brabancon_griffon": "brabancon_griffo",              # stanford -> tsinghua 철자
    "leonberger": "leonberg",                             # oxford   -> stanford 철자
    "japanese_chin": "japanese_spaniel",                  # oxford   -> stanford 철자 (동일 견종)
    "wheaten_terrier": "soft_coated_wheaten_terrier",     # oxford   -> stanford 철자
    "german_shorthaired": "german_short_haired_pointer",  # oxford   -> stanford 철자
    "scottish_terrier": "scotch_terrier",                 # oxford   -> stanford 철자

    # english_cocker_spaniel: v1에서 "__NOT_MVP__"로 막았으나 v2에서는 별도 클래스로 복원.
    # (FIX 항목을 두지 않으면 그대로 통과해 BREEDS에 매칭된다.)
}


def canon(label: str) -> str:
    b = str(label).lower().replace("-", "_").replace(" ", "_")
    b = FIX.get(b, b)
    if b in NOT_DOG:
        return "__NOT_DOG__"
    return b


def main():
    mpath = ROOT / "data" / "manifests" / "master_manifest.parquet"
    df = pd.read_parquet(mpath)

    is_cat = df.original_breed.str.startswith("CAT_")
    df["canonical_breed"] = None
    df.loc[~is_cat, "canonical_breed"] = df.loc[~is_cat, "original_breed"].map(canon)
    df.loc[is_cat, "canonical_breed"] = "__CAT_OOD__"
    df["breed_id"] = df.canonical_breed.map(BREED_ID).astype("Int64")
    # 컬럼명 in_mvp25는 하위 호환 위해 유지 (build_split / export_processed가 이 이름을 쓴다)
    df["in_mvp25"] = df.breed_id.notna() & df.usable

    df.to_parquet(mpath, index=False)

    # ---- taxonomy.csv (freeze) ----
    rows = []
    for (src, orig), grp in df[~is_cat].groupby(["source", "original_breed"]):
        c = grp.canonical_breed.iloc[0]
        if c == "__DROP__":
            action = "drop"
        elif c == "__NOT_DOG__":
            action = "not_dog"
        elif c in BREED_ID:
            action = "mvp"
        else:
            action = "not_mvp"
        rows.append(dict(source=src, original_label=orig, canonical_label=c,
                         breed_id=BREED_ID.get(c, ""), action=action, n_images=len(grp)))
    tax = pd.DataFrame(rows).sort_values(["action", "canonical_label", "source"])
    tax.to_csv(ROOT / "data" / "manifests" / "taxonomy.csv", index=False, encoding="utf-8-sig")

    # ---- 검증 출력 ----
    mvp = df[df.in_mvp25]
    counts = mvp.groupby("canonical_breed").size().sort_values()
    print("클래스 수: {} / 목표 {}".format(mvp.canonical_breed.nunique(), len(BREEDS)))
    print("usable images: {:,}".format(len(mvp)))
    print("\n--- 장수 하위 15종 (train이 50장 미만이면 prototype 품질 주의) ---")
    print(counts.head(15).to_string())

    missing = sorted(set(BREEDS) - set(mvp.canonical_breed.unique()))
    if missing:
        print("\n[!] manifest에서 한 장도 못 찾은 견종 {}종: {}".format(len(missing), missing))
    else:
        print("\n[OK] BREEDS 126종 전부 manifest에서 발견됨")


if __name__ == "__main__":
    main()
