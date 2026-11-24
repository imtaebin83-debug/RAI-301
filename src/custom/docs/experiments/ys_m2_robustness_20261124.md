# Mission 2 stopping and robustness — ys — November 24–25, 2026 (KST)

Base: `447889e` (`opt/m2-ys-20261121`); main checked at `7542e7e`.
Work is confined to Mission 2. Shared navigation, localization, vision, planner,
simulator, referee and environments remain unchanged.

## Changes and evidence

The detector, three-second stopped survey, approach speed, surface correction
and stopping tolerance are retained. The final heading behavior is tested
separately below. The previous final runtime passed 15/15
standalone runs and 2/3 Mission 1→2 runs; the remaining failure was before cone
detection. Four subsequent green approaches in Mission 2→4 also passed.
These earlier runs are separate from this report's new tests.

The new controller explicitly rejects nonfinite x, y or heading before the
final approach and at the final stop. A NaN comparison previously could bypass
the range/error checks. A fault-injection regression against the published
controller fails 8 of 12 subcases (NaN/infinity, three axes, before/after the
approach); the other four were already rejected by its range checks. The
updated controller rejects all 12 and stops. This validates
input handling, not the accuracy of finite localization estimates.

The evaluation manifest now includes all six cone orders × all three command
colors, preserving the previous case IDs. The nine added cases are
`order-3-*`, `order-4-*`, and `order-5-*`. Three earlier fresh-layout runs already
tested a subset of those combinations; this is additional regression coverage,
not a claim that all layouts are new held-out data.

CPU checks: 19 Mission 2 tests pass, including advancing-but-desynchronized
RGB/depth, unsupported depth encodings, a frozen depth stream, nonfinite poses,
and post-rotation stopping errors. The 46 existing shared tests also pass.

`perception_replay.py` exercises the actual detector on saved RGB-D frames.
Six usable frames spanning all three colors retain detection under 60%/130%
brightness, 20% missing depth, and a 2m vertical coordinate shift. The maximum
XY change is 0.01443m. All-invalid input is rejected for every capture.
In total, 41 scored checks pass. Five older captures had no baseline detection;
their 25 positive-detection variants are explicitly unscored, not counted as
passes. This is deterministic saved-sensor evidence, not new simulator success.

## Reproduction

Use the existing `haic_rai301_verify` environment, prepend the chosen worktree's
`src` to the ROS-enabled `PYTHONPATH`, and leave other users' processes alone.

```bash
python -m unittest custom.mission_2.test_mission_2 -v
python -m custom.mission_2.perception_replay --output /tmp/new-cone-replay.json /path/to/captures/*.npz
python custom/mission_2/evaluate.py --run --source-root "$PWD" --output /tmp/new-m2-run \
  --case order-3-blue --case order-3-green --case order-3-orange \
  --case order-4-blue --case order-4-green --case order-4-orange \
  --case order-5-blue --case order-5-green --case order-5-orange
```

## Initial matrix and stopping candidate

The first nine added cases ran with the nonfinite-input guard but the previous
final-turn behavior. They passed 8/9. `order-5-blue` failed in 84.49s: detection
succeeded at (17.1311, 24.6928), the final approach overshot and recovered, then
the controller rejected the post-rotation position. The official pelvis was
(17.309,24.140), versus box center (17.0,24.05), with dx=0.310m and dy=0.090m.
The shared localizer also recorded a reanchor after rejecting a 0.480m step.
This does not isolate all localization error from physical turn-induced motion.

The official `_judge_2` uses only x/y position, not heading. A separate candidate
therefore removes the unnecessary final heading command, holds the arrival
heading, waits a bounded 0.5s for settling, and retains the final estimated
position check. It records the settled pose/error on both success and failure.
No sensor pose reset, target snapping, referee input, or shared control edit is
introduced. Continuous Mission 4 validation from the changed endpoint is
reported below.

A repeat launcher was refused by the simulator lock (exit 75) because the
teammate had started Mission 6. It did not execute a mission and is not included
in the 8/9 count. The pre-stop-change baseline then passed a real repeat in 55.68s, confirming
intermittency. The no-final-turn candidate passed the same case in 58.52s, with
official absolute errors x=0.03m / y=0.01m. These single runs do not establish
a speed gain or isolate causality. The complete candidate matrix and continuous
Mission 2→4 validation are recorded below; CPU checks alone are not simulator
PASS results. Finite global drift and shared
route stalls remain outside this change's scope.


## Final standalone results

Runtime commit: `38c4a8f`. The separate targeted run and the full nine-case
matrix all passed: **10/10 standalone PASS**, with no controller errors.

| Case | Official verdict | Seconds | Absolute x / y error (m) |
|---|---|---:|---|
| order-3-blue | PASS | 51.32 | 0.081 / 0.012 |
| order-3-green | PASS | 56.89 | 0.045 / 0.019 |
| order-3-orange | PASS | 48.36 | 0.003 / 0.012 |
| order-4-blue | PASS | 48.34 | 0.020 / 0.015 |
| order-4-green | PASS | 51.70 | 0.115 / 0.032 |
| order-4-orange | PASS | 55.45 | 0.011 / 0.011 |
| order-5-blue | PASS | 54.61 | 0.061 / 0.029 |
| order-5-green | PASS | 48.07 | 0.044 / 0.014 |
| order-5-orange | PASS | 51.17 | 0.188 / 0.088 |

The additional targeted `order-5-blue` run passed in 58.52s with exact recorded
absolute errors 0.026m / 0.012m. Across all ten final-code runs, maximum absolute
x/y errors are 0.188m / 0.088m, versus a 0.25m limit per axis. Runtime source
hashes for the controller, cone detector, shared vision/navigation and localizer
match the tested local files. Shared code was not edited. Remote worktree base
SHAs alone do not identify the candidate; the records retain source fingerprints.

The initial guard-only matrix's 8/9 and its 1/1 repeat are separate diagnostic
runs, not included in the final 10/10. This finite regression batch is not a
competition-wide success probability or proof that localization is reliable.

## Continuous runs and comparison

| Case | M2 official / controller | M2 seconds | M4 result |
|---|---|---:|---|
| Green, final code | PASS / no error | 48.68 | PASS, 166.69s |
| Side blue, final code | PASS / no error | 56.01 | FAIL before inference, route timeout, 161.72s |
| Side blue repeat, final code | PASS / no error | 56.01 | FAIL before inference, pose recovery timeout, 73.94s |
| Side blue, previous final-turn behavior (comparison only) | PASS / stop-check error | 57.24 | FAIL before first observation, route timeout, 100.08s |

Thus the final Mission 2 controller passed **13/13 attempts**: ten standalone
and three continuous, all without controller errors. Mission 4 passed only the
green continuous sequence. Both blue sequences failed on the unchanged shared
route/localization dependency, before VLM inference. The old-turn comparison
also failed upstream of VLM, demonstrating that the endpoint blocker exists
without the new stopping behavior. This small comparison does not establish
that the two behaviors have equal failure probabilities.

The comparison's Mission 2 official PASS must not hide its controller error:
`Final rotation moved outside the estimated cone target`. It is a diagnostic
control, not a final-code success. A source audit confirms that the only
custom Python file changed in the comparison is `mission_2/mission_2.py`.

No new Mission 1→2 validation was run in this iteration. The previously
recorded Mission 1/shared-route blocker remains an integration dependency.

## Evidence and integration

The adjacent evidence directory contains complete official records for the
initial matrix, repeat, final matrix, targeted run, continuous failures and
comparison; CPU logs; the detector perturbation report; and source audits.
`prestop_controller.py.txt` is the exact pre-stop-change controller snapshot,
matching all nine original records. Copy it only into a separate evaluation
worktree when reproducing that comparison. Raw logs and images are retained on
OMEN under the recorded `/tmp/ys-*` paths and locally under
`output/ys-missions/optimization-20261124/`. The original 11 RGB-D captures remain
in the two `/tmp/ys-m2-*-captures` directories named by the replay report; large
sensor files and model weights are excluded from Git.

For combined reproduction, use an isolated checkout containing Mission 2
runtime `38c4a8f` and Mission 4 runtime `257965b`, preserving shared code, then
run Mission 4's evaluator with `--case after-cone` or `--case after-cone-blue`.
The latter is new evaluation coverage and is not a claim that the continuous
blue route is solved. The integration owner must still address pose drift,
reanchors and stalled progress. Main is not merged automatically.
