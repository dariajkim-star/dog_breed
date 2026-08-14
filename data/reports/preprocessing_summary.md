# Preprocessing Summary — 전처리 완료 보고 (8/14)

## Dataset Summary

| 단계 | 수치 |
|---|---|
| Raw 수집 (6소스) | 98,402장 (breed 3소스) + OIv7 21,586 + COCO 177 + MuttMix CSV |
| Corrupt/annotation 오류 제외 | 2,375장 (`usable=False`, 사유 기록) |
| Exact duplicate 제거 (keep-one) | 19,442장 (MD5 or pHash≤5 — 그룹 19,053개) |
| Taxonomy: MVP 25종 확정 | 46,497장 (teddy→poodle, chinese_rural_dog DROP, corgi 병합) |
| Split FREEZE (group-aware 70/15/15) | train 32,549 / val 6,978 / test 6,970 |

## 최종 산출물 (processed/) — 검증 완료

| 폴더 | 내용 | 수량 (기대치 대조) |
|---|---|---|
| `detection/` | YOLO 공식 포맷 + data.yaml | train 18,397 / val 1,540 / test(COCO) 177 — 이미지·라벨 쌍 ✅ 일치 |
| `breed_body/` | 25종 body crop 518px (standard_crop 15%) | 44,506장 ✅ manifest 기대치 일치, 실패 0 |
| `breed_head/` | Tsinghua head crop | 40,935장 ✅ 일치 |
| `ood/` | Oxford 고양이 (val/test 반반) | 2,400장 |

- crop 육안 QA: `crop_gallery.jpg` — 귀·꼬리 보존, 체형 왜곡 없음, 엣지는 회색 padding
- OIv7 라벨 정제: IsGroupOf/IsDepiction 3,093건 제거, 무효 이미지 1,649장 탈락

## 주요 발견/결정 기록
1. **Stanford의 90.6%(18,645장)가 Tsinghua에 포함** — pHash로 검출, keep-one 제거로 해소
2. MuttMix Dryad에 사진 없음 → 믹스견 검증은 정성 평가 + 인간 베이스라인 인용으로 전환
3. Oxford body bbox 부재 → YOLO11n pseudo-bbox (4,517장 채택 / 473장 needs_review 제외)
4. embedding 기반 dedup 생략(시간) — pHash ≤10 보수적 그룹핑으로 대체. **개선 트랙 1순위**

## 한계 (발표에서 정직하게 언급할 것)
- Tsinghua 고유분도 인터넷 출처 — ImageNet 계열 pretrain 기준 완전한 "외부" test는 아님 (DINOv2 사용으로 완화)
- 같은 개의 다른 세션 사진은 pHash로 못 잡음 (embedding dedup 미실행)
- Tsinghua 라벨 노이즈 일부 잔존 가능 (chihuahua 등 믹스 의심 샘플)
