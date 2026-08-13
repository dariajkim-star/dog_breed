[1단계]
강아지를 찾는 법 학습
Open Images
      ↓
YOLO
      ↓
Dog Detector


[2단계]
25개 견종을 구별하는 법 학습
Tsinghua + Stanford
      ↓
Breed Encoder
      ↓
견종 특징 벡터


[3단계]
각 순종의 대표 특징 생성
Golden 이미지 수백 장 → Golden Prototype
Poodle 이미지 수백 장 → Poodle Prototype
Husky 이미지 수백 장 → Husky Prototype
...


[실제 사용]
사용자 믹스견 사진
      ↓
YOLO
      ↓
강아지만 Crop
      ↓
Breed Encoder
      ↓
이 강아지의 특징 벡터
      ↓
25개 순종 Prototype과 거리 비교
      ↓
가까운 견종들을 Top-K로 출력
      ↓
유사도 Calibration / Normalize
      ↓
Golden 46%
Poodle 31%
Cocker 14%
Other 9%
