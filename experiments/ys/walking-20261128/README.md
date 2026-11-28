# Walking-policy handoff — 2026-11-28

## Status and objective

The objective is one integrated, reproducible draft of custom code satisfying rulebook v2 before members optimize individual missions. The current mission integration is functional; submission compliance and custom-policy integration are not complete.

The six-mission baseline is on **integrate/ys-all-missions-v2-20261127**, report commit **58e43ddd00afe3aa8bb219c67488d7397272465b**, tested runtime **ca9fa1f6894cc6320e9c9fc371cfc84a271f2718**. It passed M1→M2→M3→M4→M5→M6 in one official run: **889.12 scored seconds, zero respawns**. It used the supplied G1 walking policy (walking.policy is null) and the team's learned M5 keypose policy. It is one development layout, not all layouts/orders or competition certification.

See [the full six-mission report](../../../src/custom/docs/experiments/ys_all_missions_v2_20261127.md). This new handoff branch inherits that integration and adds experiments/documentation only. No deployed policy, mission code, simulator, referee, or team instructions were changed. Main was not merged.

## Latest pilot: complete, NOT ready for overnight training

Remote directory: /home/aril/rai-training/ys-walk-progress-pilot-20261128

All 500 additional iterations finished, with evaluations at 100/250/500, in **1159.48 seconds (19.3 minutes)**. Full actor, critic, normalizers and optimizer were restored from the previous refinement checkpoint 500. Final learning rate was **5e-6**, 512 environments. A first smoke test at 2.5e-5 with 64 environments regressed; it is preserved on OMEN under smoke/. The revised 512-environment smoke test is under smoke_conservative/ and passed all three short mixed-motion trials.

Final checkpoint:
- /home/aril/rai-training/ys-walk-progress-pilot-20261128/iteration_0500/checkpoint.pt
- Same directory: small_policy.onnx, export_parity.npz, simulator results, curriculum_gate.json.
- Exact hashes and earlier checkpoints: [history and checkpoint inventory](history_and_checkpoint_inventory.json).
- Current assessment: [assessment.json](snapshot/assessment.json). Final official-physics motion checks do NOT pass all readiness gates.

Final results:
- Short mixed-motion sequence: **3/3 no falls**, velocity RMSE 0.0598 m/s, maximum tilt 6.99°.
- Backward gentle restart: first burst **0.116 m**, second **0.180–0.181 m**; heading error about **11.3°**. Both distances miss the 0.20 m engineering target.
- Left: first burst **0.332–0.335 m**, second **0.186–0.222 m**; heading drift **33–34°**, above the 20° target.
- Right: first burst **0.191 m**; second **−0.094 to −0.101 m** along the original intended direction; heading drift **62°**. Wrong-way displacement after restart remains a blocker.
- Extended 64-second sequence: **fell at 9.59 seconds during −0.25 m/s backward movement**.
- These are simulator results for this checkpoint, not whole-mission results with our policy.

The previous refinement checkpoint 500 is also worth keeping:
 /home/aril/rai-training/ys-walk-restarts-refine-20261128/iteration_0500/checkpoint.pt
It had better left heading (~19°) and stronger second backward burst (~0.37 m), but weak first backward movement, ~62° right heading error, and an extended-test fall at 11.96 s. The newest checkpoint is not universally better.

## What changed in the latest training experiment

- Normalized commanded-speed error penalty, weight −0.8, penalizes standing still/wrong-way/overspeed for active straight motion.
- Smooth heading cost 2*(1−cos(error)) keeps distinguishing errors above 57°; the previous capped square saturated there.
- Two-thirds of lateral samples are rightward; one-third remain leftward. Overall sampling: 40% forward, 30% backward, 25% lateral, 5% turning.
- Preserves the earlier move–stop–same-move curriculum (3.5–4.5 s moving, 1–2 s pauses), heading reference across both bursts, and exact official stationary-idle assistance.
- No external assistance during active walking; no deployed source changes.
- Conservative fixed LR 5e-6 and stop after two consecutive mixed-motion regressions.

## Next owner: first actions

1. Check OMEN process/GPU status and exit.txt/done.json before starting anything. Do not stop teammate jobs. The latest bounded pilot is complete as of this handoff; recheck live status.
2. Read final assessment and both-burst metrics; do not choose checkpoints by training reward or iteration number alone.
3. Diagnose right-side turning and weak initial backward motion; preserve left-side/forward behavior. Compare the previous refinement checkpoint with the newest one.
4. Consider the training/evaluation speed mismatch explicitly: stage 0 samples only 0.10–0.12 m/s backward/lateral, while the extended test asks for 0.25 m/s backward and 0.20 m/s lateral. Curriculum advancement currently requires all gentle-direction gates. Do not merely lower the test thresholds or claim the faster command is trained.
5. Run bounded pilots with export/official-physics tests. Do not start an overnight run until there is repeatable useful motion, not just standing without falling.
6. After a candidate is qualified, integrate it into a separate test checkout via the documented robot.yaml walking.policy mechanism. Keep the supplied-policy baseline intact and rerun all six missions, randomized layouts, orders/transitions, and respawn behavior.
7. Package the trained weights, exact source/revisions, training configuration/seeds/logs, data/provenance and reproducible commands. Coordinate M5 data packaging and team-server LLM/VLM deployment.

## OMEN and environments

Shared Ubuntu account aril on aril-13 (Tailscale 100.77.17.47); use your own registered SSH key and keep host verification enabled. ED25519 fingerprint:
 SHA256:BOPtEgffXuc9T8q65Dbp2ysZ7dLyvQcl+Juuap5iLIk

- Training Python: /home/aril/miniforge3/envs/unitree_mjlab_omen5060/bin/python
  Python 3.11, PyTorch 2.7.1+cu128, MJLab 1.2, MuJoCo/MuJoCo-Warp 3.5.
- Competition/evaluation Python: /home/aril/miniconda3/envs/haic_rai301_v2/bin/python
  Python 3.10, MuJoCo 3.7, ONNX Runtime 1.23.2; ROS Humble for mission runs.
- Frozen competition checkout: /home/aril/rai-worktrees/ys-all-v2-20261127
- Shared exclusion lock: /tmp/rai-gpu-simulator.lock
- No OS upgrades, shared environment changes, or changes in teammates' worktrees.
- Large checkpoint binaries, raw logs, models and datasets remain on OMEN. They are not backed up by this GitHub commit. Preserve/copy the exact checkpoint files in the inventory before deleting any remote experiment.

## Continue safely in a NEW directory

The published code is an exact snapshot of the latest experiment and deliberately retains known OMEN paths. It is not a portable environment installer. protected.json verifies the working loader/model/robot configuration/source checkpoints. The source archive below includes the earlier training variants.

Use a separate checkout of this branch. Copy code/*.py and code/protected.json into a new absolute experiment directory; never run again in an existing checkpoint directory. Choose the seed checkpoint deliberately. For example, after selecting NEW_RUN as a new absolute path and SEED as one of the inventory checkpoints:

~~~bash
# Run on OMEN from this folder in your separate checkout.
nvidia-smi
mkdir "$NEW_RUN"
cp code/*.py code/protected.json "$NEW_RUN"/
cd "$NEW_RUN"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
PY=/home/aril/miniforge3/envs/unitree_mjlab_omen5060/bin/python
"$PY" test_contract.py
"$PY" test_restarts.py
"$PY" test_directions.py
"$PY" test_restart_gates.py
"$PY" test_progress_rewards.py
nohup flock -n -E 75 /tmp/rai-gpu-simulator.lock \
  timeout --signal=TERM --kill-after=30s 3600 \
  "$PY" -u pilot.py --iterations 500 --envs 512 \
  --initialize-actor "$SEED" > train.log 2>&1 < /dev/null &
~~~

The loop automatically saves/evaluates at 100/250/500, checks protected hashes, and writes progress.json/assessment.json/done.json. It raises an early-stop error after two consecutive mixed-motion regressions. The original launcher in snapshot/run.sh additionally writes exit.txt and GPU telemetry, but its hard-coded directory points to the completed run: **do not execute it unchanged**. The example above does not write exit.txt; verify done.json, early_stop.json, the log and process state. Lock exit code 75 means occupied; do not bypass it.

Smoke test changed code in a separate directory before a larger pilot. Never edit the source used by a running job.

## Model interface and known transfer limits

The competition ONNX interface is **[1,98] → [1,29]**, with normalization embedded in the export.
98 observations: root angular velocity, projected gravity, smoothed command, gait sin/cos, joint-position offsets, joint velocities and previous raw actions. Exact original joint order, default pose, gains, action scales, phase reset, slew and idle behavior are tested. No linear velocity, foot contacts or pelvis position are added to the actor inputs. Privileged simulator quantities are used for training rewards/critic and external evaluation only; whether all training/provenance details satisfy organizer interpretation remains part of final review.

Physics: 2 ms × 10 = 50 Hz control; gait period 0.6 s. Training uses the exact robot asset/mechanics on nominal flat ground. MJWarp lacks competition noslip_iterations=4 support, so training uses 0 while official evaluation keeps 4. GPU/CPU MuJoCo versions differ. Terrain/friction/load robustness remains unproven. Idle root assistance mirrors the supplied loader and must not be mistaken for learned standing balance.

Engineering gates (not rulebook scoring rules): both bursts ≥0.20 m, lateral drift ≤0.20 m, stop drift ≤0.10 m, heading error ≤0.35 rad, velocity RMSE ≤0.14 m/s, yaw-rate RMSE ≤0.25 rad/s, tilt <20°, three starting headings. The heading target is stricter than the supplied policy's ~24–31° drift in the dedicated test. Overnight assessment also requires successful short/extended checks at two checkpoints.

## Rulebook-v2 draft completion checklist

Rulebook v2 p26 requires sensor-only runtime control, local LLM/VLM on the team server (no external commercial API), team-trained humanoid AND manipulator action policies, and training code/data verification.

- [x] Six mission implementations integrated and one continuous official pass with supplied G1 walking.
- [x] Local Qwen planner tested separately; local VLM used in the mission baseline.
- [x] Existing team-learned M5 keypose policy and learning code preserved (not a successful end-to-end CNN policy).
- [ ] Qualify and integrate a team-trained G1 policy.
- [ ] Retest complete missions with that exact policy/configuration, including competition command speeds.
- [ ] Verify more randomized layouts and mission orders, respawn/recovery and recording load.
- [ ] Test full user-instruction/planner/runner flow; the prior LLM parsing check was separate from the explicit-list mission chain.
- [ ] Verify actual team GPU-server model installation/connection and final execution environment.
- [ ] Package M5 original demonstrations and G1 training provenance/artifacts for submission.
- [ ] Review final rules/hidden-rule announcements and freeze one submission revision.

M5 source data: /home/aril/m5-data/pilot; outputs /home/aril/m5-runs; successful integration trace /home/aril/rai-backups/ys-all-v2-20261127/mission5-success-trace.tar.gz. Rulebook copy used: /home/aril/rai-backups/20261127-v2/rulebook_v2.pdf, SHA256 99dfc6db883a27d9016a0c2159d0180d9e5df8d25d266e6098445c27aa6ffc13. The earlier v2 status document has superseded mission-status sections; the six-mission report and this handoff are newer.

## Included artifacts

- code/: latest trainer/configuration, export evaluators, regression tests and protected hashes.
- snapshot/: final latest-pilot metadata, tests, assessment and milestone summaries.
- history_and_checkpoint_inventory.json: earlier pilot summaries and complete-checkpoint paths/hashes.
- historical_training_sources.tar.gz: prior experiment Python sources, launchers and configuration/protection manifests; no model binaries.

