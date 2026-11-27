# RAI 301 기본 구현

**2026-11-27 Starter/Rulebook v2 반영:** 조교 `f03a500`과 팀 구현을 병합하고 `robot.yaml`, 공개 센서 API·리스폰, Panda 행동 좌표를 호환시켰다. 새 `haic_rai301_v2`에서 M1→M2→M5 연속 3/3 PASS, 회귀 검사 110개 통과. 최신 미션별 main/팀원 브랜치 구분·미통합 성과·규칙상 준비 과제는 [v2 업데이트와 진행도](docs/V2_UPDATE_AND_MISSION_STATUS_20261127.md)를 먼저 읽는다. 아래 예전 성공 기록을 v2 전 배치 검증으로 해석하지 않는다.

규칙과 원본 API를 검토한 센서 기반 초기 구현이다. 미션 1·2는 실제 시뮬레이터 PASS를 확인했다. 미션 4는 방 분류 3/3 정답이나 배달 실패이며, 미션 3·6도 아직 성공하지 못했다. 미션 5는 팀 학습 keypose 정책의 파지·몸통 검출을 안정화해 동일 배치 8개에서 기존 4/8 → 개선 8/8 PASS, 별도 새 배치에서 12/12 PASS를 확인했다. 실행·재학습은 [미션 5 안내](mission_5/README.md), 최신 증거·한계는 [파지 안정화 보고서](mission_5/STABILITY_20261122.md)에 있다. 다른 미션의 기존 결과는 `docs/VALIDATION_REPORT.md`와 `docs/validation/`을 확인한다. 팀별 작업 방법은 `docs/TEAM_OPTIMIZATION.md`에 있다.

## 설치와 실행

조교 원본에 이 `custom/`을 넣고 원본 README의 ROS 의존성·Conda 설치 스크립트를 실행한다. 추가 Python 패키지는 `requirements.txt`에 기록되어 있다. 모델 파일은 Git의 대용량 제한 때문에 Git 추적에서 제외했지만 현재 Ubuntu 작업 폴더에는 내려받아져 있다. 새 PC에서는 대회 Conda 환경에서 다음 명령으로 같은 revision을 내려받는다. 제출할 때는 실제 가중치 폴더도 `custom/models/`에 포함한다.

```bash
python custom/tools/download_models.py
```

현재 GPU 노트북에서 검증한 작업 폴더와 새 환경:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_v2
cd ~/RAI-301/src
export PYTHONDONTWRITEBYTECODE=1
python main.py --missions 1 2
```

SSH에서 화면 없이 실행할 때는 `bash custom/tools/run_sim.sh --headless --missions 1 2 --no-wait`를 사용한다.

위 명령의 GUI는 Ubuntu 화면에서 열린다. SSH 헤드리스 검증은 `MUJOCO_GL=egl`과 화면만 끈 임시 config를 사용한다. `python custom/testing/randomized_suite.py --seed 301 --cases 2 --missions 1 2`는 평가 목적으로 임시 시나리오를 생성하고 원본 main/referee를 실행한다. 같은 GPU에서 다른 시뮬레이터·학습이 실행 중이면 먼저 실행 시간을 조율한다.

```bash
python -m unittest discover -s custom/tests -v
HAIC_REQUIRE_LLM=1 python custom/planner/planner.py \
  'Go to the blue cone, then score a soccer goal.'
python main.py --llm-web
```

## 구조와 협업 경계

| 파일 | 역할 / 권장 담당 |
|---|---|
| `localizer_node.py`, `localization.py`, `navigation.py`, `bringup.launch.py` | 연속 LiDAR 위치 추정·이동. 공통 기반 담당 한 명이 관리 |
| `vision.py`, `mission_2/`, `mission_6/` | RGB-D 콘·공 인식과 접근. 물체 인식 담당 |
| `mission_3/` | 공개 틈 배치를 이용한 바닥 경로. 추락 회복·지도 개선 필요 |
| `vlm.py`, `mission_4/` | 실제 VLM 방 분류와 교실 이동. 인식·관측 자세 개선 필요 |
| `planner/` | 실제 CPU Qwen 또는 호환 API 플래너, 엄격한 JSON 검증 |
| `mission_5/` | 학습 keypose 기본 정책, 실제 시연 수집·학습·평가. 파지 안정성 개선 필요 |
| `testing/`, `tests/`, `docs/` | 평가·회귀 검사·실험 증거. 미션에서 평가 시나리오를 읽지 않음 |

각자 미션 브랜치에서 해당 파일을 수정하고 검증 결과와 함께 병합한다. `experiments/<이니셜>/`은 실험 기록에 사용하고 실행에 필요한 최종 코드는 `custom/`에 둔다. 공유 의존성, 센서 설정, 보행 상한은 통합 담당자와 조율한다.

## 실행 환경 변수

- 기본 플래너는 `models/Qwen2.5-1.5B-Instruct`를 CPU로 실행한다. API는 `HAIC_LLM_MODEL`, `HAIC_LLM_API_KEY`(또는 `OPENAI_API_KEY`), `HAIC_LLM_BASE_URL`로 설정한다. 정확한 기본 URL은 `planner/planner.py` 참조.
- `HAIC_REQUIRE_LLM=1`: 규칙 기반 개발 모드를 차단한다. `HAIC_ALLOW_RULE_PLANNER=1`은 모델 없는 오프라인 개발 전용이며 대회 검증에 쓰지 않는다.
- 기본 VLM은 `models/SmolVLM-500M-Instruct`를 CPU로 실행한다. 비전 API는 `HAIC_VLM_BASE_URL`, `HAIC_VLM_MODEL`, `HAIC_VLM_API_KEY`로 설정한다.
- `HAIC_INITIAL_POSE=x,y,yaw_deg`: 평가에서 초기 배치를 명시적으로 바꾼 경우에만 설정한다. 기본값은 README의 공개 시작 자세 `16,11,0`이다. 미션 선택 응답의 nominal 위치를 현재 위치로 사용하지 않는다.
- `HAIC_MAP_CORRECTION=1`: 공개 벽 좌표에 대한 실험적 보정. 실제 절벽 시험에서 실패해 기본값은 꺼져 있다.

KISS-ICP 위치 추정은 누적 드리프트가 있으며 전역 재위치 추정을 대신하지 못한다. 큰 추정 점프가 생기면 마지막 센서 추정 자세로 로컬 지도를 제한적으로 다시 시작하고, 반복 손실 시 정지한다. 이는 정답 좌표로 위치를 복구하는 기능이 아니다. CPU 모델 로딩은 현재 호출마다 수행하므로 미션 시간 최적화 여지가 있다.
