# Mission 4 reattempt — ys — November 22, 2026

**Status: reattempt complete; 5/6 standalone PASS; continuous operation remains blocked.** Shared localization/navigation remains unchanged.
Based on private Mission 4 commit `6a1a3af` and group main `5bfd039`; main
was checked again through GitHub and is still `5bfd039`.

## Change and limits

Mission 4 stops on the existing Navigator stale-pose exception, then waits at
most 8 seconds for advancing `/custom/estimated_pose` stamps sustained for one second, with
no observed publication gap above 0.5 seconds. Navigator
must also accept the pose as fresh. Only then may the same operation resume,
with the original deadline and at most three recovery attempts for the entire
mission. Missing poses, exhausted budgets and unrelated errors still fail.
Recovery attempts and elapsed waits are included in the mission result.
This is a mission-level response to a transient outage, not an estimator fix.
It does not repair accumulated drift or independently certify pose accuracy.

Public corridor/door targets, VLM weights and original language prompt, camera
freshness checks, room ordering and x=20 delivery target are unchanged.
If joint assignment is ambiguous, Mission 4 scores one fixed crop per captured
image: x=10%..90%, y=0%..80%. It averages each full-image/crop score pair and
reapplies the same 0.25 rejection margin. These are derived views of the same
frames, not independent observations. There is at most one retry, within the
mission deadline; persistent ambiguity still fails without entering a room.
No shared sources, settings, dependencies, OS, or other missions were changed.

## CPU checks

Final candidate: 21 Mission 4 checks passed; 46 existing checks also passed on OMEN in
`haic_rai301_verify`. New checks cover waiting stopped for advancing stamps,
permanent outages, the original deadline, retry limits and unrelated errors.

## Simulation protocol

One original simulator/referee at a time, ROS domain 74, team wrapper lock,
stock spawn (16,11,0), default map correction OFF, CPU SmolVLM. The six room
orders are previously inspected development/regression cases, not a new
untouched evaluation set. Case IDs do not control simulator scheduling;
intermittent failures need repeated runs rather than deterministic claims.

Unchanged comparison: rooms-3, previously failed at 60.53s, PASSED 164.08s on
this reattempt. Thus differences between single trials cannot establish that
the recovery code caused an improvement.

## Reproduction

Use the verified environment, own source root and the evaluator commands in
`ys/m4/DEPENDENCY_REQUEST.txt`. The final records beside this report contain
controller SHA256s, original referee outcomes, scenario and launch arguments.
Raw logs and camera images are saved under Mac workspace
`output/ys-missions/reattempt-20261122/` and the corresponding `/tmp/ys-m4-*`
directories on OMEN. Runtime controllers never read evaluation answers.

GitHub publication is requested after this reattempt, on the personal ys
branch only. No merge into main is authorized.

## Development trials (kept separate from final regression results)

| Trial | Case | Result | Seconds | Note |
|---|---|---|---:|---|
| Unchanged `6a1a3af` | rooms-3 | PASS | 164.08 | Confirms intermittency |
| Two-message retry | rooms-2 | FAIL | 180.60 | Recovered twice; third outage stopped |
| Two-message retry | rooms-3 | FAIL | 147.83 | Ambiguous VLM; no pose recovery triggered |
| Sustained-pose retry, before crop addition | rooms-2 | PASS | 200.85 | No pose recovery triggered in this pass |

The alternative A/B/C VLM prompt misclassified two saved images and was
rejected. The original prompt on fixed crops, combined with full-frame scores,
correctly resolved the saved ambiguous rooms-3 survey (joint margin 1.267). This is development
replay evidence, not an additional simulator PASS.

The read-only timing probe recorded 1760 LiDAR and 1787 pose messages. Maximum
receive gaps were 0.170s and 1.222s respectively. The probe was stopped after the
successful official run; its initial shutdown handler emitted an exception,
after flushing the final counts. This did not affect the official run. The
included evaluator-only probe handles ROS shutdown cleanly.

## Final standalone regression results

| Case | North / middle / south | Full result | Seconds | Pose recoveries | Failure |
|---|---|---|---:|---:|---|
| rooms-0 | classroom / medical / cafeteria | PASS | 209.51 | 1 | — |
| rooms-1 | classroom / cafeteria / medical | PASS | 240.05 | 3 | — |
| rooms-2 | medical / classroom / cafeteria | FAIL | 231.40 | 1 | could not enter the classroom |
| rooms-3 | medical / cafeteria / classroom | PASS | 162.34 | 0 | — |
| rooms-4 | cafeteria / classroom / medical | PASS | 214.35 | 2 | — |
| rooms-5 | cafeteria / medical / classroom | PASS | 160.82 | 0 | — |

**5/6 full PASS**, compared with 4/6 in the earlier batch. All six surveys gave
all three correct joint-assignment labels (18/18). This small repeated set
is not an estimate of general reliability. Independent image argmax labels
remain mostly medical; uniqueness-based joint assignment is not independent
classifier accuracy.

Seven recovered pose outages occurred across four final runs; three of those
runs completed delivery, while rooms-2 later stalled at entry. Recovery waits
were bounded and kept the original movement deadlines. No final standalone
run triggered the crop retry; that path is supported by the actual saved-image
VLM replay and CPU control-flow checks, not a claimed live crop-recovery pass.

Final controller SHA256: `c764a395585e479eb1b23f5dea9965d8b43f8ea7406fdeb562295f992efc5712`. Every final record matches it.
The only failed layout stopped outside the room at official (17.971,11.221),
while the local estimate was near (17.7,10.1) targeting (20,10.5). The common
localizer logged a sensor reanchor after a rejected 31.2-degree turn. A focused
shared pose-drift/heading investigation remains necessary; no common code was
changed and the experimental wall-correction option remains OFF.

## Continuous Mission 2 → 4

No reset between missions. Final Mission 2 source from private commit `15fc73a`
was combined with the final Mission 4 controller in the isolated integration
worktree. Mission 2 PASS 47.27s; Mission 4 FAIL 118.62s. The total run was 168.75s.
The mission stopped when sustained fresh localization could not be recovered.
This remains an integration blocker and is not counted as a standalone pass.
The Mission 4 branch alone does not contain the separately owned Mission 2 changes;
reproduce the continuous case in the combined checkout documented in its JSON.

All seven final scenario records match the final Mission 4 controller hash.
Full runtime logs and camera evidence remain outside Git; compact official
records, CPU logs, timing probe and owner handoff are included in this branch.

The continuous failure occurred during the middle-room survey. The shared
localizer explicitly reported `LiDAR recovery limit reached (2 per 30s); stopped
until respawn/restart`, after repeated scan-map reanchors. The mission waited
8.00s without a sustained recovery and stopped. Final official pelvis was
(11.937,11.310,0.104); the low height is consistent with a fallen robot, but the
record does not identify the contact/cause of a fall. The last navigation
estimate was approximately (17.02,14.91). The owner should inspect accumulated
position/heading error, recovery-limit health reporting, and stability during
continuous navigation; increasing mission retries cannot repair this failure.

## VLM resource profile

A final CPU-only replay of the three final rooms-0 camera images used the
unchanged SmolVLM500M worker, four Torch threads and empty CUDA_VISIBLE_DEVICES.
The worker exited 0; GNU time measured elapsed 1:01.07 and peak RSS
4,261,452KiB (4.06GiB). Model load was 0.48s;
summed image inference time was 57.43s.
This is a separate saved-image worker measurement, not total simulator memory
or an additional mission PASS. No API service or new model download was used.
