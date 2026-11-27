# Starter v2 반영·미션 진행도 — 2026-11-27

**조교 최신 코드를 팀 구현 위에 3-way 병합하고, v2 호환 변경을 `custom/`에 적용했다. 새 환경에서 미션 1→2→5 연속 공식 3/3 PASS, 회귀 검사 110개와 `pip check`를 통과했다.** 이 결과는 기본 배치 1회의 호환 확인이며, 전 미션·전 랜덤 배치 검증을 뜻하지 않는다.

기능 성공과 대회 준비 완료는 구분해야 한다. Rulebook v2 26쪽은 외부 LLM/VLM API 금지, 팀 서버의 local 모델 사용, humanoid와 manipulator 모두 팀이 직접 학습한 action policy와 학습 증빙을 요구한다. 현재 `walking.policy: null`은 제공 G1 보행 정책을 사용한다. 문구대로라면 이 상태로 대회 준비 완료라고 할 수 없으며, 제공 보행 정책의 예외 여부 확인 또는 팀 학습 정책 준비가 필요하다. M5는 실제 성공 시연에서 학습한 keypose 정책과 학습 코드를 보유하지만 원본 데이터는 아직 Git 밖에 있다.

## 1. 안전하게 반영한 범위

| 항목 | 값 |
|---|---|
| 팀 main 업데이트 전 | `7542e7efbbaf1f55c47d92a8acbc12070df37538` |
| 공통 조상 | `052a00be32fdd818550c3b2d176b8c1c1db13b7d` |
| 조교 최신 | `f03a50078bed86c8e751c5d82066dd14025d0e55` (11/26 업데이트) |
| 작업 브랜치/격리 경로 | `update/starter-v2-20261127`, `/home/aril/RAI-301-v2` |
| 백업 | `/home/aril/rai-backups/20261127-v2/team-before-update.bundle` 및 `main-before.sha` |
| 새 Conda | `haic_rai301_v2` |

`git pull`로 작업 폴더를 덮지 않았다. 먼저 main이 clean임을 확인하고 Git refs를 bundle로 백업했다. `git fetch upstream main` 후 별도 worktree에서 공통 조상을 기준으로 병합했다. 충돌한 조교 예제 미션은 팀 구현을 유지하고 새 API에 맞춰 수정했다. 팀원의 dirty worktree나 실험 프로세스는 수정·종료하지 않았다. Git bundle은 커밋 이력 백업이며 타 worktree의 미커밋 변경까지 담은 백업은 아니다.

공식 관리 영역(`custom/` 밖)은 조교 `f03a500`과 파일 내용·경로가 일치한다. 실행기·referee·물리 환경을 우리 편의대로 수정하지 않았다. `custom/` 센서·속도 값과 학습 JSON은 이전 값 그대로 보존했다. 삭제된 캐시·구형 내장 점프/킥 코드·OpenPI 클라이언트 경로 정리는 공식 변경을 따른 것이다.

## 2. 코드·규칙의 핵심 변경

| 변경 | 우리 대응/영향 |
|---|---|
| `sensors.yaml`·`walking.yaml` → `robot.yaml` | `sensors`, `walking`, `policies` 구조로 이전. 팀 설정 값 동일 여부를 자동 비교 |
| Mission의 `latest`, `last_status`, `gap_falls` 등 제거 | `custom.sensor_mission.Mission`이 허용 센서 캐시와 공개 리스폰 횟수만 제공. 금지 상태 API는 복구하지 않음 |
| `/g1/mission_status` 구독 및 조교 명령 직접 발행 금지 | 실제 6개 미션·localizer에서 사용 제거/차단. `begin_mission()`과 공식 runner/referee 경로만 사용 |
| `/g1/respawned` Empty 공개 | 위치 추정은 공개 시작점 `(16,11), 0°`로 재초기화. 직전 캐시 LiDAR를 버리고 새 scan을 기다림 |
| Panda world→base TF 제거 | 로봇 관절 FK는 유지하고, 문서가 공개한 world 기준 z축 −90° 회전만 행동 변환에 사용. 월드 위치 추정·하드코딩 없음 |
| Panda RGB의 충돌 형상·진단 site 표시 제거 | 예전 20/20을 새 환경 검증으로 취급하지 않음. 새 환경 기본 배치 연속 시험에서 M5 PASS 재확인 |
| `/panda/retry`, 개발 policy/standby 경로 제거 | 수집기의 `--retry`는 실행 전에 명확히 거부. 단일 sim 평가만 허용하고 여러 배치 시험은 fresh-process benchmark 사용 |
| TeleDex 수집·LeRobot 변환·OpenPI 소스 추가 | 공식 `tools/collect_panda.py`, `tools/convert_panda_hdf5_to_lerobot.py`, `openpi/` 수신 |
| 새 필수 pip 패키지 | 공식 스크립트로 `h5py`, `teledex==0.0.7`, `qrcode` 포함 설치·import 확인 |

`begin_mission()`은 이제 `None`을 반환한다. 이를 현재 위치로 사용하지 않는다. localizer는 별도 공개 센서 노드와 executor로 실행해 조교 `SimLink`의 상태 구독에 의존하지 않는다. TF 발행은 sim clock을 사용한다. 기존 `gap_falls` 이름은 공개 Empty 알림 수의 호환 alias일 뿐 숨겨진 판정 상태가 아니다.

리스폰은 전체 미션 합산 2회까지 허용되고 매회 총 시간에 30초가 붙는다. 골반 높이 0.30m 미만 또는 기울기 60° 초과가 5초 지속되면 최초 시작점으로 복귀한다. 복귀 시 콘 색 순서·방 배치·공 위치가 재배치되므로 기존 검출·방 분류·공 위치를 재사용하면 안 된다. 현재 baseline은 리스폰을 감지하면 해당 이동을 멈추며, 재진입·재탐색·재시도 전략은 추가 개선 대상이다.

## 3. 환경과 실제 검증

기존 `haic_rai301`을 clone한 뒤 **새 환경에서** 조교의 `setup/install_conda_env.sh`를 실행했다. 기존 Lab 6/GMR/MJLab GPU 환경은 변경하지 않았다. import 검사: MuJoCo, ONNX Runtime, OpenCV, Pillow, Shapely, SciPy, rclpy, h5py, TeleDex, qrcode 모두 성공. `pip check` 정상.

OpenPI 소스가 있다는 것과 학습 환경 설치 완료는 다르다. OpenPI의 Python 3.11+/JAX·Torch 환경은 ROS Humble Python 3.10과 분리해야 한다. README의 약 16GB짜리 `uv sync`는 기본 ROS 설치와 별도인 선택 단계로, 이번에는 실행하지 않았다. 현재 M5 keypose 실행에는 필요하지 않다.

| v2 새 환경 실제 연속 시험 | 공식 결과 |
|---|---:|
| 미션 1 | PASS, 23.8초 |
| 미션 2, green | PASS, 21.9초 |
| 미션 5 | PASS, 32.8초 |
| 합계 | 3/3, 미션 합 78.54초, 리스폰 0회, 총점 시간 78.54초 |

원본 `main.py --missions 1 2 5 --no-wait`를 화면만 끈 임시 config로 실행했다. 다른 팀의 GPU 예약이 끝날 때까지 대기했고 별도 `ROS_DOMAIN_ID=87`을 사용했다. 공통 검사 54개, M5 검사 56개, 합계 110개 통과. 공개 Empty 알림·구독 캐시·리스폰 재초기화는 모의 검사도 통과했지만, 실제 넘어짐 후 완전 재개 시험은 아직 하지 않았다.

증거: `docs/validation/v2_20261127/`의 공식 JSON·로그·설치/검사 기록·`integration_manifest.json`. 원본 백업과 설치 전체 로그는 `/home/aril/rai-backups/20261127-v2/`에 있다.

## 4. 미션별 현재 진행도

아래 팀원 성과는 별도 표시가 없는 한 **v2 반영 전 코드/환경의 기록**이다. main과 팀원 최신 브랜치를 섞어 하나의 완성본처럼 세지 않는다.

| 미션 | main 상태 | 팀 전체의 최신 성과 | 우선 개선 / 성공 후 최적화 |
|---|---|---|---|
| **1 이동** | 초기 공통 Navigator+LiDAR 위치 추정. v2 연속 시험 PASS | CSH의 IMU 보조·ICP 갱신 전 검증·health/정체 복구가 미커밋 worktree에서 진행 중. 긴 왕복 baseline FAIL187.12초 → guarded/gyro PASS130.35/129.95초, 기본 M1 PASS22.41초 기록 | 위치 추정 오염·드리프트·지연과 문 통과 정체가 최우선. 검증된 CSH 변경만 통합. 속도 상향보다 루프 왕복·다른 미션 종료점·리스폰 복구 회귀를 먼저 고정 |
| **2 콘** | 초기 검출·접근 코드. v2 green PASS | 게시 `opt/m2-ys-20261121@2284468`(runtime `38c4a8f`), 최신 공식 **13/13 PASS**=단독10+M2→4의 M2 구간3. main 미통합 | 최신 정지·신선 프레임·NaN 차단 통합 우선. 새 stop 버전의 M1→2 검증은 없음. 성공 후 3초 정지 관측과 과도한 접근 시간을 줄이고 0.5초 안정화 효과 유지 |
| **3 절벽** | 공개 경로 13점을 따르는 초기 코드. 공식 PASS 없음 | 확인된 개발 3회 FAIL, 최종53.77초는 코스 미완주·stale localization. 별도 최신 성공 브랜치 발견 못함 | 바닥/틈·발 디딤과 경로 여유, 위치 추정·team G1 policy 필요. 새 리스폰 시작점과 전역 2회 예산 반영. 준비 전에는 경기 순서 마지막에 배치 |
| **4 방·배달** | 초기 VLM+배달 코드이며 기존 최종 시험 FAIL. v2 전체 수행 미검증 | 게시 `opt/m4-ys-20261121@36d6ee5`(runtime `257965b`), 최신 단독 **3/3**, M2→4 **1/3**, 합계 **4/6 PASS**. main 미통합 | 두 blue 연속 실패는 VLM 전에 공통 경로/위치 발행 문제. M4 최신 변경만 선택 통합. VLM 모델 상주·팀 GPU 서버 연결·추론 지연 축소. 성공 시에도 실제 교실 진입 여유가 작아 위치 오차 점검 |
| **5 연필** | 학습 keypose+센서 파지 안정화 통합. v2 연속 시험 PASS | 이전 코드 고정 개발 **8/8**, 별도 seed **12/12 PASS**. 새 v2에서 20회 재검증한 것은 아님 | 새 clean RGB·리스폰·프레임 드롭/녹화 부하 회귀. 실제 미끄러짐·해제 피드백과 먼 waypoint 도달성. 학습 데이터 제출 포장. 이미 안정된 코드를 무작정 새 학습으로 교체하지 않기 |
| **6 축구** | 초기 RGB-D 공 추적·밀기, 득점 성공 없음 | 로컬 clean worktree `opt/m6-ywj-20261124@ae9a1eb`, 공식 **0/22**=baseline2+개선20. 개선20 중 일반12·외부 GT 비교 진단8 모두 FAIL. main 미통합 | 공 근거리 사각·방향 편차·위치 추정 재앵커/지연 해결. 먼 이동과 근거리 공-골대 상대 제어 분리, 짧은 밀기→후퇴/재관측. 센서 pitch 변경은 M2/M4 공통 회귀 필요 |

### 성과 해석과 증거

- M1 main 동일 baseline의 11/27 재현은 M1 2/2 PASS(17.16/21.95초), M2 2/2 PASS(27.04/23.05초)였으나 v2 적용 전이다. CSH `CSH-m1-20261127-02` 등은 계속 수정 중이므로 일부 PASS를 현재 dirty tree 전체의 검증으로 세지 않는다.
- M2 최신 단독 10개는 48.07–58.52초, 평균52.443초. 이전 버전의 15/15·M1→2 2/3은 다른 구성이다. M2→4 blue에서는 M2가 성공해도 후속 M4는 실패했다.
- M4 최신 단독 3회는 217.26/199.41/166.76초, green 연속 PASS166.69초, blue 연속 FAIL161.72/73.94초. 이전 5/6이나 저장 영상 분류 6/6을 최종 실제 성적으로 합치지 않는다. SDPA+마지막 prefix logits 최적화는 같은 저장 영상 3장의 평균84.01→63.16초로 분류를 유지하며 약25% 줄였다. 이는 전체 미션 시간 25% 감소가 아니다.
- M5 기본 정책은 성공 시연 3개에서 학습한 객체 상대 keypose+pose servo다. CNN BC 성공이나 end-to-end 두 카메라 신경망으로 부르면 안 된다. 행동에는 agentview와 자기 상태를 사용하고 wrist는 현재 기록/동기화용이다. 이전 새12개 평균33.8525초,29.87–35.64초. checkpoint SHA256 `5399000712043f249abdc3414374ddd0bf710c696a6ed3b6f623497dd7b6b3ca` 보존.
- M6 최신20회는 기본 시작점·기본 배치 단독 시험이며 랜덤/연속 성공 증거가 없다. 외부 GT 진단8회를 정상 대회 실행 성과로 세지 않는다. 구 fitting·벽/기둥 제외는 좋아졌지만 위치 발행이0.4–1.2초 이상 지연되는 경우와 근거리 시각 피드백 단절이 남았다.

주요 근거 위치:

- main: `custom/docs/validation/official/`, `custom/mission_5/docs/validation/stability_20261122/`.
- CSH: `/home/aril/rai-worktrees/CSH-m1-20261125-01/experiments/CSH/m1/`, `CSH-m1-20261127-01/experiments/CSH/m1/`, `CSH-m1-20261127-02/src/results/`.
- M2 게시 브랜치: `custom/docs/experiments/ys_m2_robustness_20261124.md` 및 같은 이름의 JSON 폴더. 조사 당시 원격 SHA `228446808d3f3c2c090428afdcc92049d2d52fe3` 확인.
- M4 게시 브랜치: `custom/docs/experiments/ys_m4_inference_20261124.md` 및 같은 이름의 JSON 폴더. 조사 당시 원격 SHA `36d6ee5fb18adbbb3900d6345e4ad20672925087` 확인. 로컬 `ys-m4-opt-20261124`의 더 오래된 dirty `c20d241`과 혼동하지 않는다.
- M6 로컬: `/home/aril/rai-worktrees/ywj-m6-20261124/src/custom/docs/experiments/ywj_m6_20261124.md`, `…/ywj/m6/20261124/runs.tsv`, `src/results/run_20261124-*.json`. GitHub 원격 게시 브랜치는 조회 시 확인되지 않았다.

## 5. 다음 작업 우선순위

1. **출전 조건부터 정리:** G1 직접 학습 정책/증빙 확보, 제공 정책 예외 여부 확인. 팀 GPU 서버 local LLM/VLM 배포·연결. 현재 shell에는 외부 모델 endpoint가 설정되지 않았고 기본은 로컬 CPU 모델이다. 외부 상용 API를 켜면 새 규칙과 충돌한다. 팀 소유 로컬 모델 서버에 HTTP로 연결하는 것과 상용 외부 API 사용을 구분한다.
2. **공통 위치 추정 안정화:** CSH 변경을 버전 고정·공유 회귀 코스로 검증해 통합. ICP 지도 오염, yaw 드리프트, 발행 주기, 문/코너 정체가 M1·2·4·6 공통 병목이다. 실패한 pose를 계속 쓰기보다 안전 정지·공개 센서 재위치추정/재관측을 설계한다.
3. **검증된 M2·M4 변경 통합:** 각 브랜치 전체를 덮어쓰지 말고 담당 runtime/검출/테스트 변경만 v2 main 위에 적용한다. M2→4 blue와 M1→2를 우선 반복한다. M5 최신 코드가 더 오래된 팀 브랜치로 되돌아가지 않게 한다.
4. **M5는 회귀·학습 증빙 담당으로 전환:** raw HDF5는 `~/m5-data/pilot`, 실험 산출물은 `~/m5-runs`에 있다. v2는 학습 코드·데이터 제출 검증을 예고하고 용량 제한이 없으므로, 원본 데이터+불변 manifest+학습 명령을 제출 경로에 별도 포장한다. 데이터가 Git 밖에 있다는 사실을 제출 완료로 간주하지 않는다.
5. **M6 MVP, M3 후순위:** M6은 짧은 닫힌 루프 밀기와 공 재관측부터 득점 1회를 만든다. M3는 현재 성공 근거가 없고 글로벌 리스폰 예산을 소비하므로 마지막에 둔다.

성공한 미션도 시간 여지가 있다: M1은 불필요 회전/우회, M2는 정지 관측, M4는 모델 cold load/추론, M5는 phase별 최소 유지 시간이다. 먼저 같은 조건의 성공률을 유지하고 줄인다. v2는 화면 녹화에 따른 frame drop도 고려하도록 명시하므로, 녹화·부하 조건 시험을 별도로 둔다.

5명 역할 예: 공통 주행·위치추정(CSH), M2/M4 통합·VLM(YS), M6(ywj), G1 학습 정책/M3, M5 회귀·데이터·전체 통합. 실제 배정은 팀 여건에 맞춘다. 우선 점수 확보 순서는 예를 들어 M5→M1→M2→M4→M6→M3이며, 당일 목표와 교실 위치·이동 거리를 보고 바꾼다. LLM에는 한 줄 영문으로 순서·당일 M1 좌표·M2 색을 모두 명시한다.

## 6. 이후 팀원이 업데이트할 때

공유 원본에서 branch 전환·reset·clean을 하지 않는다. 본인 변경을 먼저 본인 브랜치에 커밋하거나 patch로 보관하고 `git fetch origin` 후 `origin/main`을 병합한다. 충돌 시 본인의 미션 로직은 보존하되 `custom.sensor_mission.Mission`, `robot.yaml`, 공개 리스폰 처리와 새 runner/referee 규약을 유지한다. 새 main과 옛 조교 API를 섞지 않는다.

새 터미널 권장 실행:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_v2
cd ~/RAI-301/src
bash custom/tools/run_sim.sh --headless --missions 1 2 5 --no-wait
```

TeleDex 수업 도구는 Ubuntu 화면/같은 LAN의 iPhone을 준비한 뒤 `python tools/collect_panda.py`로 사용한다. 여기서는 QR 서비스나 방화벽을 열지 않았다. 팀 M5 수집기의 삭제된 `--retry` 대신 매 episode 새 시뮬레이터를 시작하거나 공식 수업 수집기를 쓴다. 다회 랜덤 검증은 `python -m custom.mission_5.benchmark --seed <seed> --cases <N> --out <새경로>`를 사용한다. 이 도구는 자체 GPU lock을 가지므로 바깥에서 중복 lock을 잡지 않는다.

## 7. Rulebook 메모

사용자 제공 PDF SHA256: `99dfc6db883a27d9016a0c2159d0180d9e5df8d25d266e6098445c27aa6ffc13`. 서버 비공개 백업에도 보관했다. 주요 근거는 16쪽(팀 학습 M5),25쪽(글로벌 리스폰),26쪽(센서·모델·정책 제한),27–28쪽(조교 노트북/팀 서버·녹화 부하),32쪽(제출)이다.

문서상 제출 마감은12/18 23:59, 대회는12/21이다. Hidden Rules 일정은20쪽에12/1,32쪽에12/4로 서로 달라 재공지 확인이 필요하다. 임의로 어느 날짜를 확정하지 않는다. 원본 코드 설명은 [공식 README, f03a500](https://github.com/goodhsm2000/HAIC-Humanoid-AI-Challenge/blob/f03a50078bed86c8e751c5d82066dd14025d0e55/README.md)을 기준으로 검토했다.
