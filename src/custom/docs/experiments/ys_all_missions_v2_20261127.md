# Integrated starter-v2 baseline: Missions 1–6

**All six missions passed in one continuous official OMEN run on 2026-11-27.** Scored time: **889.12 seconds (14m49s)**; whole run including transitions: 897.97 seconds. **Zero respawns**, zero penalties, runner exit code 0, and no mission controller returned an error. G1 was not teleported between missions; the original runner performed its normal mission initialization.

This is a verified integration baseline for the next optimization round. It is not a guarantee for every randomized layout or competition readiness.

- Integration branch: [integrate/ys-all-missions-v2-20261127](https://github.com/imtaebin83-debug/RAI-301/tree/integrate/ys-all-missions-v2-20261127)
- Frozen tested runtime: ca9fa1f6894cc6320e9c9fc371cfc84a271f2718. Subsequent report commits do not change runtime code.
- Team main base: 56f307d84934d3efc0076fa58a963ad9eb2204b2.
- Official starter: [f03a500, 11/26 v2](https://github.com/goodhsm2000/HAIC-Humanoid-AI-Challenge/tree/f03a50078bed86c8e751c5d82066dd14025d0e55).
- Main has not been merged or replaced. Team process instructions and teammates' worktrees were not edited.

## Final continuous result

| Mission | Official verdict | Seconds | Official check |
|---|---|---:|---|
| 1 — Point-goal navigation | PASS | 22.37 | 0.06 m < 0.50 m from goal world (15.50, 18.00) |
| 2 — Cone colour navigation | PASS | 35.28 | blue cone box: |dx| 0.05 < 0.25, |dy| 0.01 < 0.25 m |
| 3 — Cliff crossing | PASS | 247.88 | crossed entrance -> exit; pelvis (13.62, 11.39) inside the exit zone x 13.00..14.20, y 10.78..11.97 |
| 4 — Semantic delivery | PASS | 164.76 | pelvis (19.98, 3.59) in classroom; target classroom x 18.25..23.00, y 0.10..6.90 |
| 5 — Pencil pick-and-place (Panda) | PASS | 35.75 | pencil tip 0.144 > 0.02 m below the rim, inside the opening, released |
| 6 — Soccer goal scoring | PASS | 383.08 | ball (7.88, 1.09) inside the goal (centre x 6.82..11.68, y 0.66..1.96: whole ball past the line, between the posts) |

[Complete official record and source hashes](ys_all_missions_v2_20261127/chain-final/record.json) · [simulator log](ys_all_missions_v2_20261127/chain-final/simulator.log) · [evaluation scenario](ys_all_missions_v2_20261127/chain-final/evaluation_scenario.yaml) · [integration manifest](ys_all_missions_v2_20261127/integration_manifest.json).

The case used M1 goal (15.5,18.0), the blue cone in the rightmost slot, and rooms north-to-south medical/cafeteria/classroom. The ball used the official default placement. The Panda placement retained the default desk-relative configuration, translated into the relocated classroom. Scenario files and referee outputs belong only to the external evaluator; runtime controllers do not receive hidden placements or scoring data.

## What was integrated and changed

| Area | Implementation and provenance |
|---|---|
| Coordination and M1 | CSH's completed local v2 branch through ab6badb, including runtime a3c56be: IMU-assisted ICP prediction, validation before map updates, truthful pose timestamps/health, bounded stopped recovery, direction-preserving navigation, and a persistent local Qwen planner. This source was committed locally but had not yet been published. |
| M2 | Published owner branch at 2284468, selectively ported through 1a369e5 / integration commit 698746b. Fresh cone observations, stable stopping, finite-value checks, public sensor adapter, and respawn invalidation are preserved. |
| M3 | hskim's completed local commits 7f1e845 and e6ce01b, integrated as 4181be2/f1315f1. Added door-aware travel from the previous mission endpoint to the entrance before following the 18 public floor waypoints. Used the validated 0.28 m/s course speed and 0.12 m tolerance, with bounded initialization and execution. |
| M4 | Published branch at 36d6ee5, ported to v2 with fresh images, local VLM scoring, ambiguity handling, and respawn guards. Fixed final heading alignment by using one heading sample per command and a bounded minimum turn command; the 0.06-rad acceptance threshold was retained. |
| M5 | Preserved the working main implementation and team-trained keypose checkpoint unchanged. No policy replacement or new training was needed. The checkpoint SHA256 is recorded in the integration manifest. |
| M6 | Integrated ywj's committed local ae9a1eb ball perception and controller selectively. Ported to public sensors, replaced duplicate waypoint control with the validated Navigator, added deadline/respawn guards, backed away before post-goal observation, corrected small aiming turns, and allowed a bounded 420-second mission budget. |

M1/shared localization **did not prevent completion of any mission in the final run**. This does not mean localization is finished: M6 recorded a diagnostic counter of 8 brief localization waits, and one tight 0.06-m soccer alignment leg timed out before the controller continued and ultimately scored. There was no terminal localization failure or respawn.

The final M6 result contains a fresh sensor estimate of the ball inside the goal and no timeout/error. The official final ball position was (7.884, 1.089, 0.11), while the controller estimated (8.082,0.908). Its counters show 3 cycles, 2 completed pushes, and 1 reacquisition. These are development diagnostics, not a success-rate estimate.

## Verification

- **229 CPU checks passed:** 113 shared/integration, 60 M2/M4, and 56 M5. Logs are under experiments/ys/all-v2-20261127 at the repository root.
- The **actual local Qwen model** parsed a six-action English instruction into M1→2→3→4→5→6, retaining goal (15.5,18.0) and color blue, in 27.45 seconds. HAIC_REQUIRE_LLM=1 was set; the rule-based development planner was not used. See experiments/ys/all-v2-20261127/planner_six_missions.json.
- The simulator runs use explicit mission lists. The planner check is recorded separately and is not represented as a browser/UI integration test.
- M4 used the local CPU VLM. All three final room labels were correct.
- Official source under src/ outside src/custom/ matches upstream f03a500 exactly. Runner, referee, physics, default scenario, robot settings, and scoring thresholds were not edited.
- Every Python source hash captured when the final run started matched the source after it completed.
- The final process exited normally. The log includes a nonfatal ROS executor shutdown-thread exception after all six verdicts; that cleanup race remains a maintenance item and did not affect execution or scoring.

## Development trials retained

These trials used different runtime revisions. Do not combine them into one success-rate claim.

| Trial | Runtime | Result and interpretation |
|---|---|---|
| M6 standalone | 7022aa4 | Official PASS, 264.57s. Ball was not re-observed after the goal-mouth push; the referee confirmed the score. |
| First all-six chain | 7022aa4 | M1/M2/M3 passed. M4 recognized 3/3 rooms but timed out on final heading. M5 was invalid as a manipulation regression because the harness left its objects in the old classroom. M6 stopped short at 300s. |
| Focused M4→5→6 | cb0b915 | Official 3/3 PASS, zero respawns. M4 164.20s; M5 35.04s; M6 300.02s. M6 had scored but its controller timed out during observation, motivating the final bounded recovery change. |
| Final M1→2→3→4→5→6 | ca9fa1f | Official **6/6 PASS**, zero respawns, all controller results without errors. |

Each trial's scenario, log, source hashes, and original official result are in the matching subfolder beside this report. The evaluation-only fix translates world-frame pencil/cup coordinates by the public seven-metre room spacing when the classroom is relocated. It does not alter the referee or supply coordinates to Mission 5.

## Reproduce on the OMEN

Use a separate checkout of this integration branch. The tested checkout remains at /home/aril/rai-worktrees/ys-all-v2-20261127. The existing haic_rai301_v2 environment was reused; other environments and the operating system were not changed. Models are an ignored link to /home/aril/RAI-301/src/custom/models. A fresh machine must provision the documented models; they are intentionally absent from Git.

```bash
source /opt/ros/humble/setup.bash
source /home/aril/miniconda3/etc/profile.d/conda.sh
conda activate haic_rai301_v2
cd /home/aril/rai-worktrees/ys-all-v2-20261127/src
export HAIC_ROOT="$(dirname "$PWD")"
export PYTHONPATH="$PWD:$PYTHONPATH"
export MLSPACES_ASSETS_DIR="$HAIC_ROOT/src/environment/models/assets/molmo"
export PYTHONDONTWRITEBYTECODE=1 ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=94 YS_EVAL_ROS_DOMAIN_ID=94 MUJOCO_GL=egl
export HAIC_MAP_CORRECTION=0 HAIC_VLM_THREADS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m custom.testing.all_missions --run --case all-six-blue \
  --output /tmp/ys-six-retest-new-directory
```

Choose a new output directory for each run. The evaluator checks for GPU compute jobs and the wrapper acquires the shared simulator lock once. Exit code 75 means the lock is occupied; do not stop a teammate's process. The manifest can be inspected without starting a simulator by omitting --run and --output.

The successful M5 camera/proprioception trace is preserved only on the OMEN at /home/aril/rai-backups/ys-all-v2-20261127/mission5-success-trace.tar.gz. Large model and simulation assets were not copied to the Mac.

## Next optimization round and remaining limits

1. **M6 first:** improve contact direction, near-post ball observations, reacquisition time, and short pose waits. It is the slowest and least stable mission in these development trials.
2. **M3:** shorten the conservative cliff route time while keeping clearance and zero falls.
3. **M4:** reduce local model loading/inference time; repeat different room layouts and entry poses while preserving uncertainty checks.
4. **M1/shared:** expand long-route, sensor-delay, recording-load, and public-respawn recovery checks. The old blocking failures are not reproduced in the passing chain, but global relocalization and automatic mission restart are not proven.
5. **M2:** repeat all colors/slots with the integrated localizer before reducing observation and stopping margins.
6. **M5:** repeat randomized v2 layouts and package the original learning data/provenance. Keep the working learned policy as the comparison baseline.

The complete run is one development layout; it does not cover all randomized scenarios, all mission orders, or actual fall-and-resume behavior. Current respawn guards stop and invalidate old targets; automatic successful resumption is not established.

Runtime functionality is separate from submission compliance. walking.policy remains null, selecting the supplied G1 policy. The team's existing [v2 status report](../V2_UPDATE_AND_MISSION_STATUS_20261127.md) records the unresolved team-trained G1 policy/evidence requirement and M5 dataset packaging. This integration does not claim to resolve those submission requirements.
