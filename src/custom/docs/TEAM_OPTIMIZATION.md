# 팀 최적화 인수인계

원본 Ubuntu 저장소 `/home/aril/RAI-301/src/custom/`에 기본 코드를 통합했다. `VALIDATION_REPORT.md`의 결과를 기준으로 작업을 시작한다. 미션 5는 아직 미완료이며 스켈레톤을 유지했다.

## 5명 작업 배정 예시

| 담당 | 소유할 파일 | 먼저 해결할 일 | 완료 판단 |
|---|---|---|---|
| 통합 담당 / 본인 | localization.py, localizer_node.py, navigation.py, bringup.launch.py, planner/ | 장거리·회전 드리프트와 벽 앞 정체 개선. 미션 1·2 성공 유지 | 기존 seed301 연속 시험과 대회 모드 재현 |
| 인식·축구 담당 | vision.py, mission_2/, mission_6/ | 미션 2는 유지하고 축구공과 흰 골대/링 구분, 공 추적·밀기 개선 | 실제 공이 골라인을 넘어 공식 PASS |
| 절벽 담당 | mission_3/ | LiDAR/깊이로 바닥 가장자리 감지, 좁은 통로 제어. 공통 위치 추정 문제는 통합 담당과 수정 | 추락 횟수 제한 안에서 입구→출구 공식 PASS |
| 방·배달 담당 | vlm.py, mission_4/ | 다른 방 순서에서 VLM 분류 확인, 교실 복귀·문 통과 개선 | 방 3/3 정답과 배달을 동시에 공식 PASS |
| 미션 5 담당 | mission_5/ 및 별도 학습 코드 | 성공 시연 수집 → 팀 정책 학습 → 추론 연결부터 착수 | 별도 학습 정책·실제 Panda 판정 확인 |

## 충돌을 줄이는 규칙

한 작업 폴더에서 5명이 동시에 브랜치를 바꾸지 않는다. 각자 clone 또는 worktree를 만든다. 예를 들어 원본 저장소에서 다음처럼 시작한다. 이니셜은 실제 담당자 것으로 바꾼다.

```bash
git worktree add -b opt/m3-HSM ../RAI-m3-HSM HEAD
```

공통 기반과 `requirements.txt`, `ros_packages.txt`, `sensors.yaml`, `walking.yaml`, `bringup.launch.py` 변경은 통합 담당자가 조율한다. 미션 담당자는 처음부터 공통 이동 코드를 따로 복사하지 않고 `Navigator`를 호출한다. 위치 추정이 불안정할 때 숨은 정답 좌표로 우회하지 않는다.

실험 시제품·그림·성공률 기록은 `experiments/<이니셜>/`에 둘 수 있다. 최종 실행에 필요한 코드·모델은 `src/custom/`에 포함한다. 각 브랜치 변경은 작은 단위로 검토해 합치며, 수동으로 파일 전체를 복사해 덮어쓰지 않는다.

## 공통 인터페이스

```python
nav = Navigator(self, start)
nav.pose()                         # 센서 추정 x, y, yaw(rad)
nav.goto(x, y, tolerance=.2, timeout=100)
nav.route_to(x, y, timeout=180)     # 공개 문 좌표를 경유
nav.face(yaw, timeout=15)
nav.camera_to_world(points, frame)  # optical XYZ -> 추정 world XYZ
nav.stop()
```

`begin_mission()`은 로봇을 이동시키지 않는다. 반환 pos가 현재 위치라고 가정하지 않는다. 공통 bringup의 위치 추정기를 계속 사용하며, 오류가 나면 미션은 정지·결과 반환으로 끝낸다. 임의의 시뮬레이터 위치 재설정·randomize·judge 호출을 미션 제어에 추가하지 않는다.

## GPU 실행과 검증

공유 GPU에서 시뮬레이터는 한 번에 하나만 실행한다. 팀원 모두 아래 wrapper를 사용하면 lock으로 중복 실행을 막는다. 다른 담당자의 학습은 별도이므로 먼저 사용 시간을 맞춘다. lock 획득 실패의 종료 코드는 75다.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301
cd ~/RAI-301/src
bash custom/tools/run_sim.sh --headless --missions 1 2 --no-wait
```

GUI를 Ubuntu 화면에서 볼 때는 `--headless`를 뺀다. SSH에서 이 wrapper는 원본 설정을 변경하지 않고 화면만 끈 임시 설정을 만든다. 개인 worktree는 해당 경로에서 실행하고 필요한 모델을 `python custom/tools/download_models.py`로 준비한다.

변경 뒤에는 다음에서 관련 검사를 선택한다. 공유 GPU를 잡는 검사는 차례로 실행한다.

```bash
python -m unittest discover -s custom/tests -v
python custom/testing/randomized_suite.py --seed 301 --cases 2 --missions 1 2
python custom/testing/web_smoke.py
```

최적화 결과에는 커밋, seed/시나리오, 시작 조건, 실제 PASS/시도 횟수, 시간, 공식 판정 JSON, 실패 원인을 남긴다. 단위 검사 통과와 시뮬레이터 성공을 구분하고, 바뀐 코드에서 기존 미션이 깨지지 않았는지 확인한다. 최종 제출 전에는 원본 조교 코드에 custom만 넣어 새 환경에서 다시 검증한다.
