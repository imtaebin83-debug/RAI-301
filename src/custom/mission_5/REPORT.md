# Mission 5 구현·검증 보고서

검증일: 2026-11-21 KST. 기준 main: `646aa39629d2ac1de42da0367dc8235a6a0b1b56`.

## 결과

| 실행한 정책 | 실제 공식 referee 결과 | 의미 |
|---|---|---|
| 센서 기반 개발 교사 | 8회 중 5회 PASS | 시연 수집 결과이며 학습 정책 성능이 아님 |
| 두 카메라 CNN BC, 8 epochs | 기본 배치 FAIL, 65.8초 | 비교 실험 |
| 두 카메라 CNN BC, 100 epochs | 기본 배치 FAIL, 70.1초 | loss 감소가 폐루프 성공을 보장하지 않음 |
| **학습 keypose, 기본 배치** | **PASS, 30.4초** | 이 배치의 시연은 train에 포함 |
| **동일 keypose, 새 무작위 배치 5회** | **4 PASS / 1 FAIL** | 34.39 / 39.51 / 27.24 / 27.69 / 19.13초 |

무작위 5회는 모델을 고정하고 public `/panda/retry`로 매번 재배치한 순차 평가다. 사전에 정한 시드의 대규모 독립 시험은 아니며 80% 일반 성공률을 확정하는 결과가 아니다. 이 평가를 보고 모델·인식 threshold를 변경하지 않았다.

성공 시 공식 판정은 연필 끝이 꽂이 입구 안에 있고, rim보다 약 0.143–0.145m 낮고, 그리퍼가 놓았음을 확인했다. 마지막 실패는 `close` 단계 목표 자세에 수렴하지 못해 12초 단계 제한으로 중단했다. 공식 결과는 `outside opening`, `held=true`, 기울기 89.92°다. 삽입·해제까지 진행한 실패와 구분해야 한다. 해당 평가의 단계 trace가 없어 접촉 구속/인식 편향/자세 오차 중 근본 원인은 확정하지 못했다.

## 무엇을 학습했는가

기본 정책은 **task-parameterized keypose imitation**이다. 영상 CNN이 아니며 TP-GMM을 그대로 재현했다고 주장하지 않는다. 센서로 추정한 객체 위치·연필 yaw에 상대적인 EE 자세를 train 성공 시연에서 학습한다. phase 순서, 낮은 endpoint 분산을 만드는 기준 물체, 위치·회전 평균, 그립 부호, phase 시간 중앙값을 데이터에서 계산한다. 학습 대상 endpoint는 HDF5에 실제 기록된 자기 상태이며 교사 debug의 목표 좌표를 복사하지 않는다.

물체 상대 좌표계와 일반 SE(3) servo, 인식 알고리즘 및 gain/허용 오차는 설계한 inductive bias다. 정책 동작에는 학습된 JSON의 수치가 필요하며 교사 FSM을 호출하지 않는다. 원본 README의 '팀이 학습시킨 정책 사용' 요구에 맞춰 실제 성공 시연으로 학습한 정책을 실행한다. 특정 신경망 구조 요구는 현재 README에 없다.

개발 시연 수집에서는 agentview/wrist RGB와 관절 유래 8D 상태만 교사에 전달한다. 별도 referee가 종료 후 공식 라벨을 기록한다. 제출 정책 입력은 agentview RGB와 자기 상태이며 wrist는 현 기본 정책에서 사용하지 않는다. 로봇 FK는 공개 로봇 전용 XML을 사용한다. scenario/객체 정답/GT odom/referee 결과는 런타임 행동에 사용하지 않는다.

## 데이터와 학습 증거

총 8개 실제 episode 중 성공 5개. 고정 manifest는 마지막 실패 episode가 닫히기 전에 만들어져 7개(성공 5, 실패 2)를 기록한다. 남은 실패 1개도 원본 데이터 폴더에 보존했고 학습에는 쓰지 않았다.

| split | 성공 episode ID |
|---|---|
| train | `1795257454317550964`, `1795257703809602955`, `1795257772154559584` |
| val | `1795257804046722306` |
| test | `1795257882109485426` |

동일 센서 배치 그룹이 split을 넘지 않게 분리했다. 기본 배치는 첫 train episode다. val/test는 모델 fitting과 기준 객체 선택에 사용하지 않았다. keypose endpoint 위치 RMSE는 train 1.05mm / val 4.28mm / test 4.21mm다. 이 수치는 오프라인 endpoint 잔차이며 작업 성공률이 아니다.

CNN 비교는 동일 split에서 horizon 1, 64px, batch 32, CPU로 100 epochs 학습했다. 최선 epoch98 val loss 0.02305, 최종 train 0.000115 / val 0.02415로 과적합 격차가 있다. 실제 폐루프에서는 파지 부근 오차가 누적돼 두 CNN 모두 실패했다. 실패 CNN 가중치는 공유 노트북 `~/m5-runs/`에 보존하고 기본 제출 모델로 선택하지 않았다.

기본 `keyposes.json` SHA256:

```text
5399000712043f249abdc3414374ddd0bf710c696a6ed3b6f623497dd7b6b3ca
```

## 변경 범위·검사

- `src/custom/mission_5/`에 실행 코드, 학습 도구, 실제 학습 JSON, 문서·증거를 추가했다.
- `src/custom/requirements.txt`에 `h5py==3.15.1`을 추가했다.
- 공통 README에서 미션 5의 미구현 표기를 갱신했다. 다른 미션과 원본 `main.py`, `mission_api.py`, `referee.py`, bridge, scene, setup은 수정하지 않았다.
- 실제 Ubuntu `haic_rai301`에서 미션 5 검사 **35개**, 기존 공통 회귀 검사 **46개** 통과. `pip check` 통과.
- action translation/rotvec의 base→world 변환, robot-only FK, 데이터 누수 방지, 학습 재개·추론 export, keypose 학습값이 실제 행동을 바꾸는지 검사했다. 수치 검사는 시뮬레이터 PASS와 별도다.

공식 결과 JSON, 고정 manifest, 학습 요약·잔차는 `docs/validation/`에 있다. JSON의 referee 필드는 평가 증거이며 정책 로더는 이 폴더를 읽지 않는다. 성공 영상은 기본 배치 실제 agentview/wrist trace이며 학습 배치인 점을 함께 표시한다. HDF5 원본은 용량 때문에 Git에 넣지 않았다.

## 팀원 인수인계

실행·수집·재학습 명령은 [README.md](README.md), CNN 비교 명령은 [bc/README.md](bc/README.md)에 있다. 먼저 `close` 실패 때 trace를 수집하고 파지 중심·접촉으로 인한 자세 잔차를 확인한다. 그다음 어려운 배치의 성공 시연을 추가해 train만 확장한다. 모델 변경 전 기존 JSON을 보존하고 새 무작위 배치에서 공식 PASS/시간을 비교한다. 학습 loss만으로 모델을 승격하지 않는다.
