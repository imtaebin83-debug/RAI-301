# Mission 4 description scoring and continuous checks — ys — November 24, 2026

Base: `4972de3`, with previous runtime controller `6c17eec`. Scorer revision:
`21a6fd3`; final heading-budget revision: `d768848`. Main remains `7542e7e`.
Work is confined to Mission 4, its VLM helper and evaluation/report files.
Shared localization, Navigator, planner, camera/walking settings, original
simulator/referee, dependencies and environments remain unchanged.

## Recognition change

The original CPU SmolVLM500M scored bare room names. On the six saved final
surveys, independent predictions were correct for only 6/18 images, even though
the public one-of-each room prior recovered all six complete arrangements.

The new method asks for visible furniture/objects and scores three fixed
descriptions of classroom, medical and cafeteria contents. It uses the same
model, image pixels and cached-prefix conditional likelihood implementation.
No room layout or evaluation answer is read by the live controller.

Each worker also scores a neutral gray image at the actual input resolution.
Subtracting these measured scores reduces the descriptions' language bias.
The neutral reference is cached per resolution within that worker. No fitted
score offsets or evaluation labels are stored in runtime code. Full-size and
cropped inputs get references of their own dimensions.

Calibration changes independent labels, but mathematically cannot change the
one-of-each joint assignment or its margin: every permutation subtracts the same
sum. Description scoring is the part that changes joint decisions. Larger
numerical margins are not directly comparable with the old bare-label scorer
and are not confidence probabilities. The existing 0.25 rejection threshold,
one fixed crop retry, API backend and navigation algorithms are preserved.
The separate bounded heading-time change is described below.

Raw scores, neutral scores, calibrated scores, descriptions, prompt and image
dimensions are retained in observation provenance. Nonfinite or malformed
scores are rejected. Existing subprocess and mission deadlines still bound
model execution. Calibration adds one model image evaluation per resolution.

## Development and frozen replay validation

Development used the saved final rooms-0 survey and the earlier ambiguous
rooms-3 survey. Descriptions alone classified only 2/6 images independently,
but correctly assigned both complete surveys. Neutral-image calibration
improved independent labels to 5/6 without changing those assignments.

The prompt, descriptions and calibration method were then fixed before the
broader replay and live checks. On the six previously inspected final surveys:

| Saved survey | Original independent correct | New independent correct | New joint assignment |
|---|---:|---:|---|
| rooms-0 | 1/3 | 3/3 | correct |
| rooms-1 | 1/3 | 2/3 | correct |
| rooms-2 | 1/3 | 2/3 | correct |
| rooms-3 | 1/3 | 2/3 | correct |
| rooms-4 | 1/3 | 3/3 | correct |
| rooms-5 | 1/3 | 2/3 | correct |

**Independent labels: 6/18 → 14/18. Joint arrangements: 6/6 → 6/6.**
The new independent classifier got classroom 6/6 and medical 6/6 correct;
cafeteria was correct 2/6 and was called classroom in the other four images.
These saved images are development/regression evidence, not a new held-out set.
The previously ambiguous development survey is separate from the final rooms-3
survey; do not count it as a seventh standalone simulator success.

Today's first continuous run supplied three fresh images, excluded from prompt
development: independent labels improved 1/3 → 2/3, with the joint arrangement
correct under both scorers. These are fresh sensor captures of an already
known layout, not a wholly unseen environment. Cafeteria/classroom confusion
remains in some independent predictions; the uniqueness prior is still needed.

## Continuous baseline with the latest Mission 2

Published Mission 2 `ed2ce50` was combined with published Mission 4 `4972de3`
in `/home/aril/rai-worktrees/ys-m2-m4-20261124`. No reset between missions.

| Attempt | Mission 2 | Mission 4 | Interpretation |
|---|---|---|---|
| latest-combined-1124 | PASS 49.35s | PASS 191.18s | 3/3 rooms correct and successful classroom delivery |
| latest-combined-repeat-1124 | PASS 50.11s | FAIL 163.18s | Stalled before the final room survey; no recognition result |

The first run establishes that the latest pair can complete continuously. The
failed repeat means continuous reliability remains unresolved. Both used the
published old room scorer; do not attribute their difference to the new scorer.
The repeat returned `Could not reach room observation waypoint` at survey door
3.59. Shared code is unchanged; this is additional integration-owner evidence.

## Candidate verification

24 Mission 4 CPU tests and 46 existing shared tests passed on OMEN. Tests cover
neutral-reference reuse by resolution, independent-bias correction, joint-margin
invariance, score provenance, invalid values and overflow, plus the previous
mission/recovery/camera checks. The combined published code separately passed
37 mission tests plus 46 shared tests (83 total).

### Scorer revision, original 18-second heading budget

| Layout | Joint room labels | Independent labels | Delivery | Seconds |
|---|---|---|---|---:|
| rooms-0 | 3/3 | 2/3 | PASS | 253.29 |
| rooms-1 | 3/3 | 2/3 | PASS | 232.64 |
| rooms-2 | 3/3 | 3/3 | PASS | 203.18 |
| rooms-3 | 3/3 | 2/3 | PASS | 183.02 |
| rooms-4 | 3/3 | 2/3 | FAIL | 205.42 |
| rooms-5 | 3/3 | 2/3 | PASS | 188.22 |

**All six joint arrangements correct (18/18 labels); independent labels 13/18;
complete delivery 5/6.** All recorded custom-source hashes match scorer revision
`21a6fd3`, including the original heading controller. Results are not relabelled
as tests of the subsequent heading-budget revision.

The rooms-4 failure was `could not face room before deadline`, at classroom
return. It had already recognized all three rooms correctly. Two consecutive
stopped pose recoveries consumed 5.22s and 6.18s of the 18s heading leg; the robot
then timed out before room entry. This is a mission-owned timeout interaction
with shared pose outages, distinct from an incorrect room classification.

### Separate heading-budget revision

The fixed heading leg allowance changes from 18s to 30s for both survey and
return alignment. Recovery waits remain stopped, at most 8s each and at most
three attempts for the entire mission. Each leg still keeps its original fixed
deadline; retries do not extend it. The overall 600s mission deadline and
Navigator freshness checks are unchanged. This preserves more time to align
after the observed outages; it does not repair pose drift or permanent loss.

The final candidate passes 25 Mission 4 tests plus 46 shared tests. A new test
models two stopped recoveries consuming about 11.4s, followed by 8s of useful work: it
fits inside the 30s bound while exceeding the old 18s allowance. Combined
Mission 2/Mission 4 tests also pass (41 tests; 87 distinct tests including the
46 shared tests). This explains the budget interaction, not a guarantee that
every real timeout has this cause.

The scorer revision also completed Mission 2→4 without a reset: **Mission 2
PASS (48.84s); Mission 4 PASS (223.95s)**, all rooms correct and successful delivery.
No pose recovery was needed in that run. All recorded custom-source hashes match
Mission 2 `ed2ce50` plus Mission 4 `21a6fd3` for the overlaid room implementation.

The first targeted test of `d768848` failed in 156.66s before reaching the
classroom-return alignment: `Could not reach room observation waypoint` at
survey door 3.59. It recovered one pose outage in 2.31s, then stalled near the
estimated (15.38,4.64), targeting (16.00,3.59); official final pelvis was
(17.841,4.668,0.780). The localizer reported a rejected 29.2-degree turn and a
sensor reanchor. This attempt did not reach VLM inference or the originally
failed return-heading step. It is retained as a full mission failure.

The final `d768848` continuous run passed: **Mission 2 PASS (49.37s); Mission 4
PASS (183.80s)**, correct room arrangement and successful delivery, no reset.
It did not require pose recovery. All source hashes match Mission 2 `ed2ce50`
plus the Mission 4 `d768848` overlay. This verifies functional integration;
it does not prove that every intermittent heading failure is resolved.

The single affected-layout repeat also failed delivery (214.24s), while all
three room labels were correct. It completed heading alignment after one 3.87s
stopped recovery. Navigator then reported arrival with the last printed estimate
near (19.85,10.46), but the official final pelvis was (12.842,10.958,0.779),
outside the classroom. The mission controller returned normally; the official
referee correctly marked delivery FAIL. A 33.5-degree/0.403m rejected motion and
sensor reanchor were logged. Final log and referee positions are not synchronized
samples, but show a large disagreement that extra heading time cannot repair.

| Final `d768848` check | Recognition | Full result |
|---|---|---|
| rooms-4 first attempt | Not reached; route stalled | FAIL 156.66s |
| continuous Mission 2→4 | 3/3 correct | Both PASS: 49.37s / 183.80s |
| rooms-4 single repeat | 3/3 correct; heading completed | FAIL 214.24s; navigation/official position disagreement |

All final recorded custom-source hashes match `d768848` (plus `ed2ce50` for
the combined Mission 2). The bounded heading allowance passes CPU recovery
checks and works in the successful continuous run, but is not a demonstrated
complete fix for the affected layout. Its two official delivery failures remain
open. Do not present the earlier six-layout matrix as a full rerun of this
later heading revision or infer a success probability from these selected cases.

Across today's work, 12 simulator attempts were retained: two published-code
continuous baselines, six scorer-only standalone cases, one scorer-only
continuous case, and three final-heading checks. The model replays and resource
profile are separate CPU evidence and are not additional mission successes.

### Resource and implementation-equivalence check

After the scorer-only simulations completed, a separate production CPU worker
processed the same three saved rooms-0 images used in the replay. It exited 0,
took **78.59s**, and peaked at **4,426,284KiB RSS (4.22GiB)**. Model loading took
0.48s; summed image/calibration inference took 74.78s. The first result includes
the neutral-reference computation. Four CPU threads and existing weights were
used, with no GPU model inference, API or download.

All three production labels and all nine calibrated scores exactly matched
the frozen replay (maximum absolute score difference 0.0). This verifies the
production implementation against the experimental scorer. Calibration adds
one image evaluation to each same-resolution survey; it is an accuracy/cost
tradeoff, not a speed optimization.

## Reproduce

Activate `haic_rai301_verify` under `/home/aril/miniconda3`, select the own
worktree's `src`, prepend it to `PYTHONPATH`, and ensure the GPU is idle.

```bash
python -m unittest custom.mission_4.test_mission_4 -v
python -m unittest discover -s custom/tests -v
python custom/mission_4/evaluate.py --run --source-root "$PWD" \
  --case rooms-0 --case rooms-1 --case rooms-2 --case rooms-3 --case rooms-4 --case rooms-5 \
  --output /tmp/ys-m4-new-description-run
python -m custom.mission_4.recognition_replay --variant descriptions-neutral \
  --case-dir /tmp/ys-m4-candidate-reattempt-final-six/rooms-0 \
  --output /tmp/ys-m4-new-replay.json
```

Use new output paths. Continuous `--case after-cone` requires the combined
checkout so it includes the latest Mission 2; neither personal branch should
silently replace the other mission's implementation. All simulations use the
team wrapper lock once, stock spawn, original referee, ROS domain 74 and default
map correction OFF. Model inference is CPU-only, four threads, with existing
local weights and no API or download.

Raw logs, images and analysis stay under `output/ys-missions/continuation-20261124/`
on the Mac and `/tmp/ys-*1124*` on OMEN. Compact official records, CPU logs and
replay evidence accompany this report. Replay success is not delivery success;
all unsuccessful development and live trials are retained.
