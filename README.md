# Final Competition 미션 과제

> ## 11/26 업데이트
>
> git pull 실행 뒤 1장 ③의 `install_conda_env.sh`를 한 번 더 실행(pip 패키지 추가됨).
>
> - **`custom/robot.yaml`**: `sensors.yaml`, `walking.yaml`을 하나로 통합(`sensors:`, `walking:`, `policies:`). 값은 그대로 옮기면 됨 (3.6, 3.8, 3.9)
> - **넘어짐·리스폰**: 골반 0.30 m 미만 또는 기울기 60° 초과가 **5초** 지속되면 **최초 시작 지점 (16, 11), 0°** 로 리스폰하고 `/g1/respawned` 발행. 리스폰 1회마다 총 시간에 **+30초** (4장)
> - **G1 관절 직접 제어**: `/g1/joint_command`, `/g1/joint_gains` 토픽 추가 (5장)
> - **미션 5**: 데이터 수집 / 형식 변환 코드 및 OpenPI 패키지 추가 (3.10)

- **제공:** 시뮬레이터(미션 환경, Unitree G1 및 Franka Panda), G1 기본 보행 정책 및 센서 토픽, 미션 실행 기능, 판정·시간 측정, 각종 편의 기능(무작위 요소 변경, 로봇 위치 재설정 등)
- **구현:** `src/custom/` 안의 플래너(명령 → 미션 순서), 위치 추정·주행, 인식, 미션 6개, Panda 조작 정책 등등 
- **조교 관리:** `custom/` 밖 전부(`main.py`, `mission_api.py`, `referee.py`, `main_config.yaml`, `scenarios/`, `web/`, 시뮬레이터 등). 채점 시 조교 원본으로 실행하므로 고쳐도 반영되지 않음.

목차: 1 설치 · 2 실행 · 3 코드 구조와 작성법 · 4 미션 · 5 시뮬레이터가 주는 것 · 6 규칙(위반 = 탈락) · 7 제출

---

## 1. 설치 (새 PC, 한 번만)

Ubuntu 22.04, [ROS 2 Humble desktop](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

**⓪ Git Clone**

```bash
git clone https://github.com/goodhsm2000/HAIC-Humanoid-AI-Challenge.git
```

**① Miniconda**

`conda --version`이 되면 건너뜀.

```bash
cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && source ~/.bashrc
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   # 최신 Miniconda만
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**② ROS 2 패키지**

rosdep·colcon, `custom/` 안 자작 패키지 `package.xml` 의존성, `custom/ros_packages.txt` apt 패키지 설치.

```bash
bash <path to repository>/src/setup/install_ros_deps.sh
```

**③ Python 환경** — conda 환경 `haic_<조 이름>`

```bash
export HAIC_ENV=haic_<team_name>     # haic_ 뒤에 조 이름 (ex. haic_team1)
bash <path to repository>/src/setup/install_conda_env.sh
```

**④ 빌드**

자작 ROS 2 패키지가 있을 경우.

```bash
conda deactivate    # conda 밖에서 (base, haic 등 안 뜰 때까지)
source /opt/ros/humble/setup.bash
cd <path to repository> && colcon build --symlink-install
```

채점 PC도 같은 순서로 설치.
`custom/`만 받아서 ②③④를 다시 하므로, **사용할 패키지는 `custom/requirements.txt`·`custom/ros_packages.txt`에 전부 적혀 있어야** 함.

---

## 2. 실행

새 터미널마다 `conda activate <conda_environment>` 후 `cd <path to repository>/src`.

| 명령 | 하는 일 |
|---|---|
| `python main.py --llm-web` | **대회 방식**: 웹 페이지(http://localhost:8501, 자동으로 열림)에 명령 한 줄 → 플래너 → 미션 순차 실행 → 판정 |
| `python main.py` | 연습용: `missions.run`에 적힌 순서로 실행 (터미널에서 SPACE로 시작, `q` 종료) |
| `python main.py --missions 2 4` | 연습: 지정 미션만 |
| `python main.py --step` | 미션마다 SPACE로 시작(`n` 건너뛰기) |
| `python main.py --sim-only` | 개발용: 시뮬레이터(+bringup)만 띄워 두기 |
| `python custom/mission_2/mission_2.py` | `--sim-only` 상태에서 미션 하나만 실행·판정 |
| `python custom/planner/planner.py "Go to the blue cone, then score a goal."` | 시뮬레이터 없이 플래너만 |
| `python environment/robots/control/g1/teleop_wasd.py` | 키보드로 G1 조종(W/S/A/D/Q/E) — `/cmd_vel` 감 잡기 |
| `python main.py --scenario scenarios/x.yaml` | 다른 시나리오 파일로 |

Ctrl+C 한 번으로 MuJoCo 창까지 모두 종료(끄는 중 한 번 더 누르면 강제 종료).

**개발 흐름**

- 터미널 1: `python main.py --sim-only`로 시뮬레이터 실행.
- 터미널 2: 미션 파일을 반복 실행하며 개발.
- G1 위치 재설정: MuJoCo 창에서 숫자 키 `N`으로 해당 미션 시작 지점에 배치. 임의 위치는 `G`를 누른 뒤 바닥 더블클릭.
- 리스폰: `B` 키를 누르면 최초 시작 지점 (16, 11), 0°로 리스폰. 넘어진 것과 똑같이 리스폰 횟수에 포함.

**`src/main_config.yaml`**

조교 관리, 대회 당일 조교 파일 사용.
개인 PC에서는 화면 설정만 변경 권장.

| 항목 | 뜻 |
|---|---|
| `missions.run` | 연습 실행의 미션 순서 |
| `missions.wait_for_space`, `step` | 연습 실행 시 SPACE 대기 (최초 실행/이후 미션) |
| `missions.falls_allowed` | **실행 전체**(모든 미션 합산)에서 허용되는 리스폰 횟수. 넘어짐, 틈 추락, 뷰어 키 `B`가 모두 1회로 계산. n+1번째에는 리스폰 직후 진행 중인 미션이 실패로 끝나고, 남은 미션은 실행되지 않고 실패 처리. 허용 횟수 안의 리스폰도 1회마다 총 시간에 **+30초** 페널티 |
| `scenario.file` | 시나리오 파일. 대회 당일 조교 배부 파일 공통 사용 |
| `referee.*` | 성공 판정 기준 |
| `randomize.*` | `R` 키 randomize 시 미션 1 목표(복도)·공(축구장) 배치 범위 |
| `instruction.web_port`, `open_browser` | streamlit llm 페이지 포트, 자동 열기 |
| `simulator.view`, `shadow_size` | MuJoCo 창 / 그림자 해상도(GPU 메모리 부족하면 4096 또는 2048) |
| `simulator.start_pose`, `ball_friction`, `extra_args` | G1 시작 자세, 공 마찰, 시뮬레이터 추가 옵션 |

---

## 3. 코드 구조와 작성법

### 3.1 전체 흐름

```
main.py 파이프라인
   └─ main.py ─ custom/planner/planner.py  plan(instruction) ──► [(2, {"color": "blue"}), 6, (1, {"goal": [x, y]})]
                 │
                 ├─ (시작 시 한 번) ros2 launch custom/bringup.launch.py   ← SLAM·위치 추정·Nav2 등 "계속 돌 노드"
                 │
                 └─ 미션마다: custom/mission_N/mission_N.py의 MissionN() 생성 ─► run() ─► referee가 판정 ─► 다음 미션
```

### 3.2 폴더

```
src/custom/                       ★ 제출물 전부
  planner/planner.py                plan(instruction) -> 미션 순서
  bringup.launch.py                 모든 미션 동안 돌 노드들
  robot.yaml                        센서 배치, 걷기 정책·속도 상한, 사용할 정책 파일 목록
  mission_1/ … mission_6/mission_N.py   class MissionN(Mission): run()
  requirements.txt, ros_packages.txt    쓰는 pip / apt 패키지 전부
  <자유>                              보조 모듈, 지도, 모델 파일, 자작 ROS 2 패키지(package.xml)
src/main.py, mission_api.py, referee.py, main_config.yaml, scenarios/, web/   조교 관리
src/ros2_bridge/, environment/, setup/                                          시뮬레이터 및 환경 설치 (수정 금지)
src/tools/                        collect_panda.py(미션 5 시연 데이터 수집), convert_panda_hdf5_to_lerobot.py(수집 데이터 -> LeRobot)
src/openpi/                       OpenPI 소스(체크포인트·가상환경 없음). 미션 5 정책 학습·실행용 참고 (3.10)
```

**`custom/mission_N/mission_N.py`의 `class MissionN`, `custom/planner/planner.py`의 `plan`, `custom/bringup.launch.py`** 위치·이름·형태만 지키면 나머지는 `custom/` 안에서 자유롭게 수정 가능.

### 3.3 미션 파일 작성법

받은 틀의 뼈대는 그대로 두고 `__init__`과 `run()` 내부 구현.

```python
import time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from mission_api import Mission, MissionResult, run_standalone

class Mission2(Mission):                            # 이름·MISSION·run() 시그니처는 그대로
    MISSION = 2

    def __init__(self) -> None:
        super().__init__()                          # 노드 mission_2 + 시뮬레이터 API
        self.image = None
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image,
                                 qos_profile_sensor_data)      # 센서 best-effort QoS

    def on_image(self, msg):                        # run()이 도는 동안 백그라운드에서 호출
        self.image = msg

    def run(self) -> MissionResult:
        self.begin_mission(2)                       # 미션 시작 (G1은 선 자리 그대로)
        color = self.params.get("color") or self.m2_color   # 대회: 명령에서 / 연습: 시나리오 값
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:          # 종료 조건은 직접 구현
            self.cmd_vel(0.3, 0.0, 0.0)             # 10~20 Hz로 계속
            time.sleep(0.05)
        self.stop()
        return self.result()                        # 끝. 성공/실패는 referee가 판정

if __name__ == "__main__":
    raise SystemExit(run_standalone(Mission2))
```

- **`run()` 실행 시 반드시 결과 반환** 미션별 시간 제한이 없어 반환이 없을 경우 모든 미션이 정지. -> 모든 대기 (센서 수신, 서버 연결 등)에 상한 시간 설정 권장.
- `run()`이 돌아온 순간의 위치로 판정. 미션이 끝날 때 **멈춰 서 있어야** 함.
- 반환은 `self.result()` 하나. 미션 4만 `self.result(rooms=["medical", "classroom", "cafeteria"])`(북→남 방 이름) 추가로 반환.

### 3.4 플래너 작성법 (`custom/planner/planner.py`)

```python
def plan(instruction: str) -> list:
    ...  # LLM에 물어 미션 순서와 인자를 뽑기 (프롬프트 설계가 핵심)
    return [(2, {"color": "blue"}), 5, 6, (1, {"goal": [14.2, 7.7]})]
```

- 미션 번호(1–6)를 순서대로. 명령이 값을 준 미션은 `(번호, {...})`로 — 그 dict가 그 미션의 `self.params`. 키: 미션 1 `"goal": [x, y]`(월드 좌표), 미션 2 `"color": "green"|"blue"|"orange"`.
- **대회 모드(`--llm-web`)에서는 시뮬레이터가 목표·색을 알려 주지 않고,**(`self.m1_goal`, `self.m2_color`가 `None`). 명령에서 읽어 넘겨야 함. (시나리오 파일 기준으로 판정.)

### 3.5 bringup (`custom/bringup.launch.py`)

`generate_launch_description()`의 `LaunchDescription([...])` 안에 모든 미션 동안 돌아야 하는 노드(SLAM/위치 추정, Nav2, 인식 노드) 작성. 시뮬레이터가 뜬 뒤 한 번 launch되고 `main.py`가 끝날 때 함께 종료.

### 3.6 센서 배치 (`custom/robot.yaml`의 `sensors`)

| 항목 | 뜻 | 기본 |
|---|---|---|
| `sensors.g1.camera.pitch_deg`, `forward_m`, `up_m` | 머리 D435i 카메라의 아래쪽 기울기, 마운트 홈에서의 앞/위 이동 | 15, 0, 0 |
| `sensors.g1.lidar.up_m` | Ouster LiDAR 장착 위치 | 0.0 |
| `sensors.panda.agentview.pos`, `quat`, `fovy` | Panda 3인칭 카메라(베이스 좌표, w x y z) | 책상을 내려다보는 자리 |
| `sensors.panda.wrist.pos`, `quat`, `fovy` | Panda 손목 카메라 | 그리퍼 위 |

### 3.7 패키지 추가

| 종류 | 적는 곳 | 적용 방법 |
|---|---|---|
| Python | `custom/requirements.txt` | `HAIC_ENV=haic_<team_name> bash src/setup/install_conda_env.sh` |
| apt/ROS | `custom/ros_packages.txt` | `bash src/setup/install_ros_deps.sh` |
| 자작 ROS 2 패키지 | `custom/<패키지>/` + `package.xml` | 위 스크립트 실행 후 → `colcon build` |

파일 경로는 상대 경로로 작성. ex. `Path(__file__).resolve().parent / "maps" / "x.yaml"`
지도·모델 파일도 `custom/` 안에 배치.

### 3.8 걷기 정책과 속도 상한 (`custom/robot.yaml`의 `walking`)

어떤 정책을 어떤 상한으로 돌릴지 결정.
`main.py`가 시뮬레이터를 켤 때 이 항목을 읽어 `--policy`, `--max-vel`로 넘김.

| 키 | 기본 | 뜻 |
|---|---|---|
| `walking.policy` | `null` (제공 정책 `environment/robots/control/g1/walking/g1_29dof_velocity.onnx`) | 바꿔 쓸 ONNX 파일. **`custom/` 기준 상대 경로**(예: `models/my_walk.onnx` → `custom/models/my_walk.onnx`). 파일이 없으면 `main.py`가 시작하지 않습니다 |
| `walking.max_vel.vx` / `vy` / `wz` | 0.6 / 0.4 / 0.7 | 시뮬레이터가 `/cmd_vel`을 자르는 상한 (m/s 전진, m/s 옆, rad/s 회전) |

- 속도 상한 자유롭게 조정 가능. 단, 넘어지면 최초 시작 지점으로 리스폰되고 리스폰 횟수 count
- 정책을 바꾸려면 제공 정책과 **같은 입출력**이어야 함.
  29-DoF 속도 정책의 관측은 베이스 각속도, 중력 방향, 명령 vx vy wz, 보행 위상, 관절 위치−기본값, 관절 속도, 직전 action이며, 출력은 29개 관절 목표.
  정확한 순서와 스케일은 `ros2_bridge/tools/g1_walk_policy_29dof.py` 확인. 입출력이 다르면 시뮬레이터가 작동하지 않음.
- 확인: `python main.py --sim-only`로 켜고 `teleop_wasd.py`로 teleop.

### 3.9 그 밖의 정책 파일 (`custom/robot.yaml`의 `policies`)

걷기 외에 쓰는 정책 파일(킥, 점프 등)을 `이름: 경로`로 적는 곳. 미션별로 나누지 않고, 쓰는 파일을 그냥 나열.

```yaml
policies:
  kick: models/my_kick.onnx      # custom/ 기준 상대 경로 (파일 또는 폴더)
```

- `main.py`가 시작할 때 파일이 있는지만 확인(없으면 시작하지 않음).
- 미션 코드에서 `self.policies["kick"]`으로 절대 경로(`pathlib.Path`)를 받아 직접 불러서 실행.
- G1을 정책으로 직접 움직이려면 `/g1/joint_command`로 관절 목표를 보냄(5장 명령 토픽).

### 3.10 미션 5 시연 데이터 수집 (`tools/collect_panda.py`)

iPhone의 TeleDex 앱으로 Panda를 직접 조종해 시연을 녹화. 미션 5와 같은 장면·카메라·state/action 형식으로 저장되므로 정책 학습 데이터로 사용 가능.

- 준비: iPhone에 TeleDex 앱 설치, PC와 같은 Wi-Fi/LAN 연결. 방화벽 사용 시 `sudo ufw allow 8888/tcp`.
- 필요한 pip 패키지(`h5py`, `teledex`, `qrcode`)는 1장 ③에서 함께 설치됨.

```bash
conda activate haic_<team_name>
cd <path to repository>/src
python tools/collect_panda.py        # 저장: custom/mission_5/dataset/episode_000000.h5, 000001, ...
```

1. 터미널에 나오는 QR 코드를 앱으로 찍거나 `IP:8888`을 직접 입력.
2. 화면이 위를 보게 폰을 잡으면 그 자세가 기준이 되고 녹화 시작. 폰을 움직이면 그리퍼가 따라 움직이고 앱의 토글이 그리퍼를 여닫음.
3. 성공하면 **Button A**(저장), 실패하면 **Button B**(버림). 둘 다 팔을 원위치시키고 연필·연필꽂이를 새 위치에 배치.
4. 종료: 터미널 Ctrl-C 또는 MuJoCo 창 `Q`. 다시 실행하면 번호가 이어짐.

| 조작 | 동작 |
|---|---|
| 폰 Button A / B | 성공으로 저장 / 버림, 다음 과제 |
| MuJoCo 창 `S` / `F` | 성공 / 실패로 저장, 다음 과제 |
| MuJoCo 창 `Q` | 종료 |

**OpenPI (`<path to repository>/src/openpi/`)**

정책을 학습시키거나 policy 서버를 띄워 볼 때 쓸 수 있는 OpenPI 소스.

```bash
cd <path to repository>/src/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync                      # 한 번만: 가상환경 생성 (uv 필요, 약 16 GB)
# ① 수집한 에피소드(.h5) -> LeRobot 데이터셋. 성공으로 저장한 에피소드만 사용
uv run ../tools/convert_panda_hdf5_to_lerobot.py ../custom/mission_5/dataset --repo-id local/mission5_panda
# ② 학습 (config는 src/openpi/src/openpi/training/config.py에 직접 추가)
uv run scripts/compute_norm_stats.py --config-name <config 이름>
uv run scripts/train.py <config 이름> --exp-name <실험 이름>
# ③ policy 서버
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=<config 이름> --policy.dir=<체크포인트 폴더>
```

---

## 4. 미션

좌표는 모두 **world 좌표**(m, 각도는 +x 기준 반시계 °) 기준.
미션은 `self.begin_mission(N)`으로 시작하고, G1은 직전 미션이 끝난 곳에서 출발 (시작 지점까지 가는 것까지 미션에 포함).
기본 배치는 `scenarios/default.yaml`, 대회 당일 랜덤으로 변경.

**넘어짐과 리스폰.** 어느 미션에서든 G1의 골반이 0.30 m 아래로 내려가거나 60° 넘게 기운 상태가 5초 이어지면(넘어짐, 틈 추락 모두 이 기준 하나로 판단) 시뮬레이터가 최초 시작 지점 (16, 11), 0°로 리스폰. 뷰어 키 `B`도 같음. 리스폰될 때마다 `/g1/respawned`(`std_msgs/Empty`)가 한 번 발행되므로, 이 토픽을 구독해 위치 추정을 (16, 11), 0°(world)로 다시 초기화. 리스폰과 함께 콘 색 배치, 방 배치, 공 위치가 다시 뽑힘. 미션 1 목표, 미션 2 색, 연필과 연필꽂이 위치는 바뀌지 않음. 이미 판정이 끝난 미션의 결과는 그대로이고, 진행 중이거나 남은 미션은 새 배치로 판정. 리스폰 횟수는 실행 전체에서 누적되며 2회를 넘으면 진행 중인 미션이 실패로 끝나고 남은 미션도 실행되지 않음. 리스폰 1회마다 총 시간에 **+30초** 페널티.

| # | 미션 | 시작 지점 (연습 시) | task | 성공 (시뮬레이터 gt 위치 기준) |
|---|---|---|---|---|
| 1 | 지점 이동 | (16.00, 11.00), 0° | 주어진 좌표까지 걸어가 멈추기. 목표는 복도(x 14–17, y 1–20) 안 임의 지점 | pelvis가 목표 0.5 m 이내 |
| 2 | 콘 색 찾기 | 콘 방 문 앞 (15.50, 20.15), 90° | 주어진 색의 콘을 센서로 찾아 그 앞으로 이동. 콘 3개의 색 배치 랜덤. | pelvis가 목표 콘 중심에서 문 쪽(−y)으로 0.75 m 떨어진 점을 중심으로 한 0.5 × 0.5 m 사각형 안. |
| 3 | 절벽 코스 | 코스 입구 앞 (13.75, 20.125), 180° | 입구 문으로 들어가 틈 5개를 지나 출구 문으로 나오기. 떨어지면 최초 시작 지점 (16, 11)로 자동 리스폰. | 입구→출구로 통과했고 골반이 출구 앞 x 13.0–14.2, y 10.775–11.975 안. |
| 4 | 방 알아내기 + 배달 | (16.00, 11.00), 0° | 동쪽 세 방(교실·보건실·식당)이 어느 자리인지 알아내고(배치 랜덤, VLM 사용) 연필을 교실로 가져가기. 답: 방 이름을 **북→남 순서**(y 17.5 → 10.5 → 3.5; 시작점에서 +x를 보면 왼쪽→정면→오른쪽)로 반환. | 세 방을 모두 맞혔고 골반이 교실 안(문을 지나 x 18.25 이상). |
| 5 | 연필 꽂기 | Panda 책상 옆 (22.25, 19.55), 180° | Panda로 책상 위 연필을 연필꽂이에 넣기. 연필 및 연필꽂이 위치 랜덤 → Panda 카메라로 찾기. 팀이 학습시킨 정책 사용. | 연필 끝이 연필꽂이 입구 안에서 테두리보다 2 cm 이상 내려가 있고, 그리퍼가 놓은 상태. |
| 6 | 축구 | 축구장 문 (13.75, 9.10), 180° | 공을 찾아 골에 넣기(차기·밀기 자유). 공 위치 랜덤(흰 고리 위). | 공 전체가 골라인을 넘어 그물 안(기둥 사이, 크로스바 아래). |

**판정 — `referee.py`**

- `run()`이 돌아오면 G1을 멈추고 1초 뒤 **정답 위치**로 검사. 지나간 것은 인정되지 않음.
- 결과는 터미널과 `src/results/run_<YYYYMMDD>-<HHMMSS>.json`에 PASS/FAIL·근거·시간으로 기록.
- **총 시간 = 미션 시간 합 + 리스폰 횟수 × 30초.** 리스폰(넘어짐, 틈 추락, 뷰어 키 `B`) 1회마다 30초 페널티. 결과 요약과 결과 파일의 `total`(`respawns`, `penalty_seconds`, `total_seconds`)에 기록.

미션 4 판정 예시:

```
[referee]   rooms     truth  : north classroom | middle medical | south cafeteria
[referee]             answer : north medical   | middle classroom | south cafeteria
[referee]             1/3 rooms right  [FAIL]
```

### 좌표표 (월드, m)

모든 길이와 좌표의 단위는 **m**. 범위는 각 축의 **최솟값부터 최댓값까지**이며, 점 위치는 x와 y를 따로 표기.

**공간 범위**

| 공간 | x 최솟값 | x 최댓값 | y 최솟값 | y 최댓값 |
|---|---:|---:|---:|---:|
| 건물 | 5.5 | 23 | 0 | 26 |
| 복도 | 13 | 18 | 0 | 21 |
| 절벽 방 | 5.5 | 13 | 10.5 | 21 |
| 축구장 | 5.5 | 13 | 0 | 10.5 |

**주요 지점과 문 위치**

| 지점 | x | y | 비고 |
|---|---:|---:|---|
| G1 시작 | 16 | 11 | 방향 0° |
| 북쪽 방 중심 | 20.5 | 17.5 | |
| 북쪽 방 문 | 18 | 17.5 | |
| 중간 방 중심 | 20.5 | 10.5 | |
| 중간 방 문 | 18 | 10.5 | |
| 남쪽 방 중심 | 20.5 | 3.5 | |
| 남쪽 방 문 | 18 | 3.59 | |
| 콘 방 문 | 15.5 | 21.0 | 폭 1.4 |
| 왼쪽 콘 중심 | 14.0 | 24.8 | 색 배치 랜덤 |
| 가운데 콘 중심 | 15.5 | 24.8 | 색 배치 랜덤 |
| 오른쪽 콘 중심 | 17.0 | 24.8 | 색 배치 랜덤 |
| 절벽 방 입구 문 | 13 | 20.125 | 폭 1.2 |
| 절벽 방 출구 문 | 13 | 11.375 | 폭 1.2 |
| 축구장 문 | 13 | 9.1 | 폭 1.5 |

**절벽 틈 — 입구에서 출구 방향 순서**

| 틈 번호 | x 최솟값 | x 최댓값 | y 최솟값 | y 최댓값 | 틈 폭 (y 방향) |
|---|---:|---:|---:|---:|---:|
| 1 | 6.65 | 13 | 19.25 | 19.60 | 0.35 |
| 2 | 5.5 | 11.85 | 17.60 | 18.10 | 0.50 |
| 3 | 6.65 | 13 | 15.80 | 16.45 | 0.65 |
| 4 | 5.5 | 11.85 | 13.90 | 14.65 | 0.75 |
| 5 | 6.65 | 13 | 11.90 | 12.75 | 0.85 |

모든 틈의 깊이는 **1.5 m**.

**축구 골대와 공**

| 항목 | 축 | 위치 또는 최솟값 | 최댓값 |
|---|---|---:|---:|
| 골대 중심 | x | 9.25 | — |
| 골대 중심 | y | 1.4 | — |
| 골라인 | y | 2.13 | — |
| 기둥 안쪽 | x | 6.82 | 11.68 |
| 크로스바 높이 | z | 1.93 | — |
| 그물 뒷면 | y | 0.66 | — |

공 반지름은 **0.11 m**.

**성공 영역**

- 미션 2: 목표 콘 중심에서 **y를 0.75 m 줄인 점**을 중심으로 하는 **0.5 × 0.5 m** 정사각형.
- 미션 4: 교실 안에서 **x는 18.25부터 23까지**, **y는 교실 중심에서 ±3.4 이내**.
- 미션 6: 공 중심의 **x는 6.82부터 11.68까지**, **y는 0.66부터 1.96까지**, **z는 1.93 이하**.

직접 mapping한 맵은 맵을 시작한 곳이 원점이므로 `world -> map` 관계는 직접 정의해야 함. 리스폰은 순간이동이라 오도메트리가 모르니 리스폰 지점 (16, 11), 0°로 위치 추정 다시 실행.

서 있는 G1의 골반은 약 13° 앞으로 기울어 있으므로 IMU 중력 방향 이용해 2D 맵 수평 조절.

---

## 5. 시뮬레이터가 주는 것

### 토픽 (`qos_profile_sensor_data`로 구독)

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/lidar/points` | `sensor_msgs/PointCloud2` | 머리 위 Ouster(기본 OS1-32), frame `lidar_link`, 10 Hz |
| `/camera/color/image_raw`, `/camera/depth/image_raw` | `sensor_msgs/Image` | D435i 컬러(rgb8) / 깊이(32FC1, m), frame `camera_color_optical_frame` |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | 카메라 내부 파라미터 |
| `/imu`, `/camera/imu`, `/lidar/imu` | `sensor_msgs/Imu` | 골반 IMU(100 Hz), 카메라·LiDAR IMU (자이로·가속도, 자세 없음) |
| `/joint_states`, `/panda/joint_states` | `sensor_msgs/JointState` | G1 / Panda 관절 |
| `/tf`, `/tf_static` | | G1 링크와 센서 프레임(루트 `pelvis`). Panda 받침의 월드 위치는 제공하지 않음 |
| `/clock` | | 시뮬레이션 시간(`use_sim_time`) |
| `/g1/respawned` | `std_msgs/Empty` | G1이 최초 시작 지점 (16, 11), 0°로 리스폰될 때마다(넘어짐·틈 추락·키 `B`) 한 번 발행 |
| `/panda/agentview/image_raw`, `/panda/wrist_camera/image_raw` | `sensor_msgs/Image` | Panda 카메라 2대(256×256) |

| 명령 토픽 | 타입 | 내용 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | G1 보행: linear.x 전진, linear.y 왼쪽 (m/s), angular.z 반시계 (rad/s). 0.5초 안에 다음 명령이 없으면 정지 |
| `/g1/joint_command` | `sensor_msgs/JointState` | G1 관절 목표(`name` + `position` rad, 관절 이름은 `/joint_states`와 같음). 이 명령이 들어오는 동안 걷기 정책은 멈추고 29개 관절 전체가 명령 목표를 따름(`name`에 없는 관절은 넘겨받은 순간의 각도 유지). 마지막 명령 후 0.2초(시뮬레이션 시간)가 지나면 걷기 정책이 다시 동작 |
| `/g1/joint_gains` | `sensor_msgs/JointState` | `/g1/joint_command` 사용 중 적용할 PD 게인(`name`, `position` = kp, `velocity` = kd). 보내지 않으면 걷기 정책의 게인 사용 |
| `/panda/joint_command` | `sensor_msgs/JointState` | Panda 관절 목표(`name` + `position`, 관절 이름은 `/panda/joint_states`와 같음) |
| `/panda/gripper_command` | `std_msgs/Float64` | 그리퍼 벌림(m) |
| `/panda/eef_delta_command` | `std_msgs/Float64MultiArray` | 말단 이동 7개 값 (dx dy dz droll dpitch dyaw grip). 각 값은 −1~1로 정규화: 이동 1.0 = 5 cm, 회전 1.0 = 0.5 rad(회전 벡터). **월드 좌표축 기준**이며 Panda 받침은 월드에서 z축으로 −90° 돌아가 있음. grip < 0 열기, 그 외 닫기 |

`/odom`, `map`, `odom` 프레임 제공하지 않음 — 위치 추정 구현.
`main.py --ground-truth`는 정답 `/odom`을 내는 **디버깅용**이고 채점 실행에는 없음.

### 미션 API (`self.` — `mission_api.py`)

```python
self.begin_mission(n)      # 미션 n 시작: 물체 초기화(미션 6의 공 등), G1은 그대로. 시작 지점 좌표는 아래 좌표표 참고
self.cmd_vel(vx, vy, wz)   # 걷기 (계속 보내기)        self.stop()
self.create_subscription(타입, topic, 콜백, qos_profile_sensor_data)   # 토픽 구독 (rclpy 노드 그대로, 3.3 예시)
self.pose()                # (x, y, yaw): TF map -> pelvis (여러분의 TF)
self.policies              # robot.yaml의 policies: {"이름": Path}
self.params                # 플래너가 넘긴 값 {"goal": [x, y]} / {"color": "blue"} (대회 모드)
self.m1_goal, self.m2_color   # 연습 전용: 시나리오의 목표/색. 대회 모드에서는 None
self.result()              # run()의 반환값 (미션 4: self.result(rooms=[...]))
self.randomize()           # 연습 전용: 새 랜덤 시나리오 (키 R)
self.log("...")
```

### MuJoCo 창 키

| 키 | 기능 |
|---|---|
| `1`–`6` | G1을 미션 시작 지점으로 spawn |
| `G` | 누른 뒤 바닥 더블클릭 → 그 지점으로 |
| `B` | 최초 시작 지점 (16, 11), 0°로 리스폰 |
| `R` | 새 랜덤 시나리오(목표·콘·방·공·연필 전부) |
| `J` | 성공 영역(초록) 표시/제거 |
| `K` | 랜덤 생성 구역(연한 파랑·책상 위 구역) 표시/제거 |
| `F` | 좌표계(월드 원점, G1 시작점)와 문·콘 중심 좌표 표시/제거 |

각종 영역 표시는 시각화만 되고 센서에는 보이지 않음.

### 시나리오 파일 (`src/scenarios/*.yaml`)

시나리오마다 바뀌는 것 — 미션 1 목표, 미션 2 색, 콘 순서, 방 순서, 공 위치, 연필·연필꽂이 위치.
**대회 당일 조교가 뽑아 모든 조가 같은 파일로 실행**하고 `main_config.yaml`의 `scenario.file`에 기록.
연습 때는 뷰어의 `R` 키로 새 랜덤 시나리오를 뽑을 수 있음.

---

## 6. 규칙 — 위반 시 탈락

1. **명령에서 읽어야 할 것을 파일에서 읽지 않는다.** 대회 모드에서 미션 1 목표와 미션 2 색은 명령에서 플래너가 뽑아 `self.params`로 넘긴 값만 사용. `main_config.yaml`·`scenarios/*.yaml`을 열어 읽으면 위반.
2. **센서로 알아내야 할 정답을 읽거나 코드에 적지 않는다.** 콘 배치, 방 배치, 공·연필·연필꽂이 위치.
3. **미션 코드에서 `self.randomize()`를 부르지 않는다.** 연습 전용 기능.
4. **조교용 토픽을 쓰지 않는다.** `/g1/mission_status` 구독 금지, `/g1/mission_end`·`/g1/randomize`·`/g1/respawn`·`/g1/mission_begin` 직접 발행 금지(`self.begin_mission()`만 사용). 5장의 토픽 표에 있는 것만 사용.
5. **`--ground-truth`의 gt `/odom`을 사용하지 않는다.**
6. **`custom/` 밖을 고치지 않는다.** 고쳐도 채점 때 조교 원본으로 교체.
7. 미션 4는 VLM을 사용한다. 미션 5는 팀이 학습한 정책을 사용한다.

**실행 시 주의사항**

아래 항목은 탈락 사유는 아니지만, 실행 오류나 미션 실패로 이어질 수 있음.

- `run()`에서 결과를 반환하지 않음.
- 미션 파일에서 `rclpy.init/spin/shutdown` 또는 `main()` 사용.
- 터미널 입력 대기.
- 의존성 목록에 없는 패키지 사용.
- 절대 경로 사용.
- API 키를 코드에 직접 작성.

---

## 7. 제출

제출물은 **`src/custom/` 폴더 하나**.

- 포함: 플래너, bringup, 미션 6개, 정책·모델 파일, `robot.yaml`, `requirements.txt`, `ros_packages.txt`, 자작 패키지.
- 제외: `build/`, `install/`.
- 필요한 환경 변수(API 키 이름 등)와 실행 전 준비는 `custom/README.md`에 작성.

채점 PC에서는 조교의 `src/`에 제출한 `custom/`만 넣고 환경 설정 후 `python main.py --llm-web`으로 대회 시나리오를 실행.
