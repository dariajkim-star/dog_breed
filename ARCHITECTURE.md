# Code Architecture

README의 모델 아키텍처와 코드 책임을 1:1에 가깝게 맞춘 구조다.

```text
main.py
  ↓
cli.py                  # argparse schema
  ↓
commands.py             # CLI 출력/명령 handler
  ↓
pipeline.py             # stage orchestration
  ├─ preprocessing.py   # image load / standard crop / split iteration
  ├─ detection.py       # Stage 1: YOLO dog detector
  ├─ encoder.py         # Stage 2: DINOv2 encoder
  ├─ prototype.py       # Stage 3: prototype generation
  └─ scoring.py         # Stage 4~5: similarity / OOD / calibration

evaluate.py             # README 7 metrics
constants.py            # shared output constants
model.py                # 기존 import 호환용 re-export only
```

## 원칙

1. `main.py`는 실행 제어만 한다.
2. stage 구현은 서로 직접 뒤섞지 않고 `pipeline.py`가 조립한다.
3. CLI 출력과 계산 로직을 분리한다.
4. `preprocessing.py`의 standard crop 규칙은 prototype/inference에서 동일하게 사용한다.
5. 평가 로직은 inference 출력 포맷과 분리한다.
6. 반복 추론/평가 시 detector와 encoder를 주입해 모델 재로딩을 피할 수 있다.
