"""Oxford pseudo body bbox를 master_manifest의 body_bbox 컬럼에 병합.

배경:
    build_manifest.py는 Oxford의 VOC xml을 head ROI로 읽어 head_bbox에만 넣는다
    (Oxford 원본에 body bbox가 없기 때문). oxford_pseudo_bbox.py가 YOLO11n으로
    body bbox를 따로 만들어 oxford_pseudo_bbox.parquet에 저장했지만, 그 값이
    master_manifest의 body_bbox로 들어가는 단계가 없었다.

    그 결과 export_processed.py의 export_breed()가 `row.body_bbox is not None`을
    보고 Oxford 이미지를 전부 건너뛰어, Oxford에만 존재하는 견종은
    breed_body crop이 0장이 된다 (= prototype을 만들 수 없는 유령 클래스).

정책:
    - needs_review=False(conf>=0.5, 검출 1건 이상)인 것만 채택
    - needs_review=True 473장은 body_bbox를 비워둔다.
      usable은 건드리지 않으므로 head crop에는 계속 쓰인다 (v1과 동일한 취급).
    - 이미 body_bbox가 있는 행은 절대 덮어쓰지 않는다 (Stanford/Tsinghua 보호)

사용:  python scripts/merge_oxford_bbox.py
출력:  data/manifests/master_manifest.parquet 갱신 (body_bbox 컬럼)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "master_manifest.parquet"
PSEUDO = ROOT / "data" / "manifests" / "oxford_pseudo_bbox.parquet"


def main():
    df = pd.read_parquet(MANIFEST)
    pseudo = pd.read_parquet(PSEUDO)

    accepted = pseudo[~pseudo.needs_review]
    bbox_by_id = dict(zip(accepted.image_id, accepted.pseudo_body_bbox))

    before = int(df.body_bbox.notna().sum())

    # 대상: oxford 소스 & 아직 body_bbox가 없는 행만
    target = (df.source == "oxford") & df.body_bbox.isna()
    n_filled = 0
    for idx in df.index[target]:
        bbox = bbox_by_id.get(df.at[idx, "image_id"])
        if bbox is None:
            continue
        # xyxy absolute pixel — manifest 내부 표준과 동일 (build_manifest.py 참조)
        df.at[idx, "body_bbox"] = list(map(float, bbox))
        n_filled += 1

    after = int(df.body_bbox.notna().sum())
    df.to_parquet(MANIFEST, index=False)

    print(f"pseudo bbox 파일: {len(pseudo):,}행 (채택 {len(accepted):,} / needs_review {len(pseudo) - len(accepted):,})")
    print(f"body_bbox 보유 행: {before:,} -> {after:,}  (+{n_filled:,})")

    ox = df[df.source == "oxford"]
    print(f"\noxford {len(ox):,}행 중 body_bbox 보유: {int(ox.body_bbox.notna().sum()):,}")
    if "in_mvp25" in df.columns:
        m = df[df.in_mvp25]
        no_body = m.groupby("canonical_breed").body_bbox.apply(lambda s: int(s.isna().sum()))
        empty = m.groupby("canonical_breed").body_bbox.apply(lambda s: int(s.notna().sum()))
        empty = empty[empty == 0]
        print(f"body crop이 0장이 될 견종: {len(empty)}종 {list(empty.index)}")
        print(f"\nbody_bbox 없는 이미지가 많은 견종 top 5:\n{no_body.sort_values(ascending=False).head().to_string()}")


if __name__ == "__main__":
    main()
