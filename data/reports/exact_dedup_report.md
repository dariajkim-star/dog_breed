# Exact Dedup Report (keep-one)

- 기준: MD5 동일 OR pHash Hamming <= 5
- 정책: 그룹당 1장 유지, 유지 우선순위 oxford > stanford > tsinghua (데이터 적은 소스)
- 중복 그룹 수: 19,053
- 제외된 이미지: 19,442

## 소스별 제외

|          |   count |
|:---------|--------:|
| tsinghua |   19158 |
| stanford |     208 |
| oxford   |      76 |

- data/labeled 에서 제거된 파일(이미지+라벨): 30,514
- raw 원본은 보존, manifest에 exclusion_reason='exact_duplicate' + dup_keep_id 기록