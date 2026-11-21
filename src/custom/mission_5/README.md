# Mission 5 — 연필 집어서 연필꽂이에 넣기

`mission_5.py`의 기본값은 동봉한 **팀 학습 keypose 정책 `keyposes.json`**이다. Ubuntu 원본 시뮬레이터·referee로 기본 배치 PASS(30.4초), 새 무작위 배치 5회 중 4회 PASS를 확인했다. 기본 배치는 학습 시연에 포함됐다. 5회 결과는 작은 표본이며 성공을 보장하지 않는다. 상세 증거와 한계는 [REPORT.md](REPORT.md)에 있다.

## 바로 실행

현재 공유 GPU 노트북에서:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301
cd ~/RAI-301/src
unset M5_CHECKPOINT M5_TRACE_DIR
bash custom/tools/run_sim.sh --headless --missions 5 --no-wait
```

Ubuntu 화면에 GUI를 띄우려면 `--headless`를 뺀다. SSH GUI는 Ubuntu 디스플레이 설정이 별도로 필요하다. `run_sim.sh`는 `/tmp/rai-gpu-simulator.lock`으로 동시 실행을 차단한다. 종료 코드 75면 다른 사용자가 GPU 시뮬레이터를 사용 중이다. 다른 팀원의 프로세스를 종료하지 않는다.

새 환경은 원본 README의 ROS/Conda 설치를 마친 뒤 `custom/requirements.txt`도 적용한다. 이번에 추가한 패키지는 `h5py==3.15.1`이며 시연 저장·재학습에 사용한다. 기본 JSON 정책의 추론에는 Torch와 GPU가 필요 없다. 원본 환경의 MuJoCo·SciPy·OpenCV·NumPy는 필요하다.

## 구조와 규칙 경계

| 파일 | 역할 |
|---|---|
| `mission_5.py` | 원본 Mission API 진입점, 시간 제한·정지·오류 처리 |
| `ros_io.py` | 허용 RGB·관절 센서, 로봇 FK, base→world 행동 변환 |
| `perception.py` | 보정된 agentview RGB에서 연필·꽂이 위치와 연필 방향 추정 |
| `keypose_policy.py`, `keyposes.json` | 학습한 물체 상대 목표 자세·그립·시간과 일반 pose servo |
| `train_keyposes.py` | 성공 시연의 실제 측정 상태에서 keypose 학습 |
| `collect.py`, `vision_teacher.py` | 개발 전용 시연 수집·센서 기반 교사 |
| `evaluate.py` | 개발 전용 무작위 초기화·공식 referee 평가 |
| `bc/` | 두 RGB 카메라 CNN BC 비교 실험. 현재 기본 정책 아님 |

기본 행동 정책 입력은 **agentview RGB와 Panda 자기 상태**다. wrist RGB는 저장·동기화하지만 현재 keypose 행동에는 사용하지 않는다. 자기 상태는 관절 센서와 공개 로봇 전용 MJCF의 FK로 계산한 base 좌표 EE 위치·rotvec·손가락 관절 2개, 총 8차원이다. 행동은 base 좌표 `delta_xyz/.05m, delta_rotvec/.5rad, grip` 7차원이고 ROS 경계에서 world 좌표로 회전한다.

런타임은 시나리오, 숨겨진 물체 좌표, ground-truth odom, referee 결과를 읽지 않는다. 카메라 공개 보정과 로봇 설명만 사용한다. 초기 센서 인식 후 학습된 객체 상대 자세를 실제 관절 피드백으로 추종한다. 교사 FSM을 기본 정책 대신 호출하는 fallback은 없다. 학습한 수치가 없는 경우 실행을 거부한다. `custom/` 밖의 조교 코드는 수정하지 않는다.

`/panda/retry`와 referee 접근은 개발 수집·평가 프로세스에만 있다. 제출 `Mission5.run()` 및 정책은 초기화나 판정을 요청하지 않는다. `collect.py`·`evaluate.py`를 대회 미션으로 실행하지 않는다.

## 재현·재학습

공유 GPU를 예약한 뒤 터미널 A에서 본인 시뮬레이터만 실행한다:

```bash
bash custom/tools/run_sim.sh --headless --sim-only
```

같은 환경·같은 저장소 `src/`의 터미널 B:

```bash
# 새 배치마다 반복. 성공/실패 모두 저장하고 공식 PASS만 학습에 사용한다.
python -m custom.mission_5.collect --out ~/m5-data/new-batch --mode teacher --retry
# Ubuntu 데스크톱에서 수동 수집은 --mode manual. P 완료, Q 중단.

python -m custom.mission_5.bc.prepare \
  --data-dir ~/m5-data/new-batch --out ~/m5-runs/new-manifest.json
python -m custom.mission_5.train_keyposes \
  --manifest ~/m5-runs/new-manifest.json --out ~/m5-runs/new-model/keyposes.json

# candidate를 검증한 뒤에만 기본 keyposes.json을 교체한다.
M5_CHECKPOINT=$HOME/m5-runs/new-model/keyposes.json \
python -m custom.mission_5.evaluate --episodes 5 --out ~/m5-runs/new-model/eval.json
```

평가 전에 위 시뮬레이터 하나만 실행 중이어야 한다. 평가 명령은 별도 시뮬레이터를 시작하지 않는다. 평가 결과에 있는 물체 판정은 학습 정책에 전달하지 않는다.

`prepare`는 episode/센서 배치 그룹을 나누고 SHA256·카메라 설정을 고정한다. 동일 배치의 프레임을 train/val/test에 섞지 않는다. 원본 HDF5와 같은 이름의 JSON(events)이 keypose 학습에 둘 다 필요하다. 수동 시연은 phase events가 없으므로 그대로 keypose 학습에 넣을 수 없다. CNN에는 사용 가능하다. 기존 manifest를 덮어쓰지 말고 새 버전을 만들며, 기존 모델 비교에서는 val/test를 고정한다.

현재 실험 데이터는 `~/m5-data/pilot/`, 학습 산출물은 `~/m5-runs/`에 보존했다. 대용량 HDF5는 Git에 포함하지 않았다. 추론에는 동봉 JSON만 있으면 된다. 이 JSON은 train 시연 3개의 측정 endpoint에서 학습했으며 phase 순서·기준 물체 선택·목표 자세·그립·시간을 포함한다. 일반 servo gain/허용 오차와 색상 인식은 수동 설계 부분이다.

## 조절과 다음 개선

- `M5_CHECKPOINT`: 비교할 JSON keypose 또는 `.pt` CNN. 기본은 동봉 JSON.
- `M5_MAX_SECONDS=90`: 시뮬레이션 시간 제한. `M5_WALL_TIMEOUT=180`: 실제 시간 제한.
- `M5_TRACE_DIR=/절대경로`: 5개 명령마다 RGB·wrist와 상태/오차를 기록. 평가 케이스마다 별도 폴더 사용.
- `M5_EXECUTE_HORIZON=1`: 기본 매 관측 재계획. keypose는 한 action만 출력한다.

우선 파지 단계 실패를 trace로 재현하고 위치/각도 오차·색상 중심 편향을 확인한다. 그다음 어려운 배치의 성공 시연을 추가한다. 현재 모델은 lift 이후 연필 미끄러짐을 직접 추적하지 않으며 HSV 인식은 조명·색상 변경에 취약하다. 카메라 pose·해상도·제어주기를 바꾸면 보정과 데이터를 함께 갱신한다. 새 평가를 보고 수정했다면 그 평가를 개발 세트로 취급하고 최종 비교는 새 배치에서 수행한다.

```bash
python -m unittest discover -s custom/mission_5/tests -v
python -m unittest discover -s custom/tests -v
```
