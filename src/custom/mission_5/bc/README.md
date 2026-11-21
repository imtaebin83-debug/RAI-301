# 미션 5 학습 재현: 기본 keypose 정책과 선택 CNN baseline

**기본 배포 정책은 `custom/mission_5/keyposes.json`을 사용하는 learned-keypose 정책이다. 이 `bc/`의 raw 두 카메라 CNN은 선택 연구용 baseline이며, 기본 삽입 시험에 실패했다.** CNN 학습 가중치는 제출 기본 모델로 배포하지 않고 GPU 노트북의 연구 artifact 경로에만 보존한다.

기본 정책의 학습기는 `custom/mission_5/train_keyposes.py`, 추론기는 `keypose_policy.py`, 정책 선택은 `policy.py`에 있다. 기본 learned-keypose는 agentview RGB와 로봇 자기 상태를 사용하며 wrist RGB는 선택 입력이다. 아래 raw CNN은 두 RGB를 모두 입력한다. 두 방법의 결과와 입력을 혼동하지 않는다.

## 확인된 실제 결과

- 센서 기반 teacher 수집: 8회 중 공식 PASS 5회, FAIL 3회. 성공 시연 5개를 train3/val1/test1로 분리했다. teacher의 성공률은 학습 정책의 성공률이 아니다.
- Raw CNN: 8 epoch와 100 epoch 모델 모두 기본 배치의 실제 폐루프 삽입 시험에서 FAIL. 낮은 validation loss를 실제 성공으로 해석하지 않는다.
- 기본 learned-keypose: 기본 배치 PASS(약30.4초), 새 랜덤 배치 5회에서 4 PASS/1 FAIL. 적은 횟수의 현재 관측 결과이며 일반 성공률을 보장하지 않는다.

## 기본 learned-keypose 재학습

명령은 대회 환경을 활성화한 뒤 저장소의 `src/`에서 실행한다. 경로는 현재 GPU 노트북의 실제 artifact 위치다. 새 clone에는 원본 시연 데이터가 포함되지 않으므로 데이터를 확보하거나 새로 수집해야 한다.

```bash
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301
cd /home/aril/RAI-301-m5/src
python -m custom.mission_5.train_keyposes \
  --manifest /home/aril/m5-runs/manifests/pilot-v3-1795257934095983009.json \
  --out /home/aril/m5-runs/keyposes-reproduced.json
```

위 출력은 비교용 artifact이며 기존 기본 `keyposes.json`을 덮어쓰지 않는다. 학습기는 manifest의 train episode만으로 phase별 **측정된** 로봇 말단 자세, 그리퍼, 시간 prototype을 맞춘다. 물체 상대 좌표의 reference도 train endpoint 분산으로 선택한다. 수집 event의 임의 `debug target_xyz`를 학습 목표로 사용하지 않는다. val/test는 학습 후 residual 보고에만 쓰고 prototype·reference 선택에 쓰지 않는다.

최종 기본 실행은 `M5_CHECKPOINT`를 설정하지 않았을 때의 `Mission5` 경로다. 다른 모델을 비교할 때만 환경 변수로 별도 artifact를 지정한다. 실행·GPU 예약·공식 판정 절차는 미션 5 상위 문서를 따른다. `train_keyposes`가 기록한 residual도 실제 삽입 성공을 대신하지 않는다.

## 고정 데이터 manifest

- 데이터: `/home/aril/m5-data/pilot/`
- 불변 manifest: `/home/aril/m5-runs/manifests/pilot-v3-1795257934095983009.json`
- manifest SHA-256: `d6de4cbc0122d829565f3f42005db141f523758ff72163ceada324eac55dd052`
- Train: `episode_1795257454317550964.h5`, `episode_1795257703809602955.h5`, `episode_1795257772154559584.h5`
- Validation: `episode_1795257804046722306.h5`
- Test: `episode_1795257882109485426.h5`

기본 배치의 시연은 train에 포함했다. 따라서 기본 배치 rollout은 처음 보는 배치의 평가가 아니다. 최종 랜덤 rollout은 학습 시연과 구분해 기록한다. 시연을 더 수집해 새 manifest를 만들 때 기존 평가 그룹을 train으로 섞지 않는다.

HDF5는 `observation/image`, `observation/wrist_image`, `observation/state`, `actions`(또는 `action`), `time`을 사용한다. 영상은 각각 256×256 RGB uint8, 자기 상태는 Panda-base의 8D(eef xyz3+rotvec3+finger qpos2), action은 정규화된 base-frame 7D다. 감사 도구는 수집 중 `.recording.h5`를 건너뛰고 성공 라벨과 명시된 공식 실패의 모순을 거부한다. 초기조건 그룹은 split을 넘지 않는다. 정답 diagnostics는 사용되더라도 분할 그룹을 정하는 데만 쓰이고 정책 입력에는 들어가지 않는다. 학습은 HDF5 SHA를 확인해 감사 후 파일 변경을 거부한다.

## 선택 연구용 raw CNN 재현

Raw CNN은 두 카메라 CNN의 spatial softmax와 로봇 자기 상태를 입력으로 action을 회귀한다. train split의 정규화 통계만 사용하고, rotvec는 모델 내부에서 연속 rotation6D로 바꿔 ±π 분기 불연속을 줄인다. 외부 상태 계약은 계속8D다. 출력은 Panda-base의 H×7 action이며 ROS 어댑터가 실제 명령의 좌표계와 제어 주기를 맞춘다.

실제 실험은 CPU, horizon1, image64, batch32, workers0, threads4, AdamW LR1e-4였다. 먼저8 epoch를 실행한 뒤 **동일 manifest·run**에서100 epoch까지 재개했다. 아래 `RUN`에는 기존 artifact를 덮어쓰지 않는 새 디렉터리를 지정한다.

```bash
RUN=/home/aril/m5-runs/my-new-cnn-run
python -m custom.mission_5.bc.train \
  --manifest /home/aril/m5-runs/manifests/pilot-v3-1795257934095983009.json \
  --out "$RUN" --device cpu --workers 0 --horizon 1 --image-size 64 \
  --batch-size 32 --threads 4 --epochs 8 --lr 1e-4
python -m custom.mission_5.bc.train \
  --manifest /home/aril/m5-runs/manifests/pilot-v3-1795257934095983009.json \
  --out "$RUN" --device cpu --workers 0 --horizon 1 --image-size 64 \
  --batch-size 32 --threads 4 --epochs 100 --lr 1e-4 --resume "$RUN/last.pt"
python -m custom.mission_5.bc.export --checkpoint "$RUN/best.pt" --out "$RUN/inference.pt"
```

실제 연구 artifact는 `/home/aril/m5-runs/bc-pilot-v3-1795257934095983009/`에 남아 있다. `pilot_epoch8_best.pt`, `best.pt`, `last.pt`, `inference_best_epoch98.pt`, `metrics.jsonl`, `training_summary.json`을 보존했다. 기본 정책으로 자동 선택하지 않는다.

100 epoch에서 train loss는0.000115, val loss는0.02415였고, val 기준 best는98 epoch(0.02305)였다. 9→100 epoch 재개는226.96초, peak RSS 약658MiB였다. 큰 train/val 격차와 실제 기본 배치 FAIL이 있어 추가 데이터·모델·폐루프 제어 연구가 필요하다. validation은 모델 선택용이며 해당 숫자를 새 배치 성공률로 표현하지 않는다.

`best.pt`는 val loss 기준, `last.pt`는 재개용이다. v3 checkpoint에 모델·정규화·manifest SHA·FPS·action frame이 있고 학습 checkpoint에는 optimizer/scaler/RNG도 포함된다. export는 optimizer/RNG만 제거하고 동일한 추론과 provenance를 보존한다. CPU `Policy`는 `M5_TORCH_THREADS` 기본2, 최대8로 제한한다. 테스트는 `python -m unittest custom.mission_5.tests.test_learning -v`로 실행하며 데이터 누수·재개·export 행동 동일성을 확인한다. 이 검사는 실제 삽입 성공 검사가 아니다.
