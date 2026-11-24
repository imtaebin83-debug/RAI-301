# Mission 4 CPU inference optimization — ys — November 24–25, 2026 (KST)

Base: `c20d241` (`opt/m4-ys-20261121`); runtime commit: `257965b`.
Main checked at `7542e7e`, including after the connection interruption.
Only Mission 4's VLM and evaluation tools change. Shared movement, localization,
planner, sensor settings, environments, model weights and dependencies do not.

## Optimization

The classifier only uses the final prefix token's vocabulary scores. On models
that explicitly expose `logits_to_keep`, it now requests that token alone.
Candidate-answer continuation still scores every answer token, with a separate
copy of the prefix cache per candidate. Models without that interface keep the
previous full-logit path.

The deployed Idefics3/SmolVLM uses the installed CPU SDPA attention path. Other
user-selected architectures retain eager attention. The actual chosen backend
is included in each observation. Images, descriptions, neutral-image calibration,
one-of-each assignment and rejection threshold are unchanged.

On the same three saved images, the initial serial comparison of model load
plus survey inference measured 82.63s
and 4.119GiB peak RSS for eager/full-logit inference, versus 65.57s and 3.242GiB
for SDPA/final-prefix-logit inference: about 21% less time and memory. Four CPU
threads and the existing local model were used. Another user's Mission 6 was
running during both measurements, so these are matched-input measurements on
a shared machine, not isolated hardware throughput guarantees. The largest
calibrated-score change was 0.0000248, with identical labels and assignment.
A second comparison in reverse order measured 60.74s / 3.242GiB for the
optimized path and 85.40s / 4.240GiB for eager/full logits. Across the two
pairs, time fell by 21–29%; the two-run mean fell from 84.01s to 63.16s
(about 25%). Labels and assignments again stayed identical.

The production loader initially exposed an argument mismatch when passing an
explicit config; this was corrected to the supported `attn_implementation`
argument before simulator tests. The failed CPU attempt is retained in the
local raw evidence. No environment upgrade was needed.

## Recognition experiment

A fixed cafeteria description naming food and a coffee machine was tested on
two development surveys (`rooms-0` and `rooms-3`). Independent classification
remained 4/6, and both joint assignments remained correct. The candidate was
rejected; the shipped descriptions are unchanged. No empirical score offsets,
hidden room order, scenario reads or referee truth enter the controller.

The actual production loader/scorer also passed a replay of all six saved
surveys (18 images). All independent labels and all six joint assignments are
unchanged; independent accuracy remains 13/18 and joint room labels 18/18.
Maximum score difference is 0.0000515. The smallest joint runner-up margin
in these six surveys is 2.5303, above the unchanged 0.25 rejection threshold;
this margin is not a probability or an accuracy guarantee. Load-plus-survey
times range from
60.63 to 64.90 seconds. These saved frames were already used in earlier
regressions and are not a new held-out recognition dataset.

Using the existing fixed crop as the primary image also remained 4/6 on the
same two development surveys. Averaging full/crop scores did not help either.
Both assignments stayed correct, but there was no recognition gain, so neither
change is deployed. The crop remains available only for the existing bounded
ambiguous-assignment retry.

An in-memory dynamic-int8 trial over all Linear layers reduced the three-image
development survey to 26.93s, but its memory peak was 3.569GiB (10.10% above
SDPA float32, narrowly beyond the predeclared 10% gate). It also changed the
previously correct medical image to cafeteria, while correcting the cafeteria
image; independent accuracy stayed 2/3 and the joint assignment was correct.
This candidate is not deployed. Stored model weights were never modified.

The narrower text-decoder-only int8 trial took 57.06s and 3.464GiB, but
independent accuracy fell to 1/3 (the cafeteria became medical). Its joint
assignment remained correct. It also missed the predeclared 15% speed-gain
threshold against SDPA float32. Both quantization trials are rejected; the
production model remains float32.

The first live final-code run, `rooms-0`, passed all three room labels and
delivery in 217.26s. The actual production observations record SDPA and a
combined model-load/inference time of 59.39s. No pose recovery was needed.
The complete standalone batch passed 3/3, covering a classroom in each of the
north, middle and south slots. These are three live orders, not a new six-order
live matrix. The six-order check above is saved-image recognition replay.
A faster inference path does not resolve the known
cafeteria/classroom confusion or shared localization failures.

## Reproduction

In the existing OMEN environment, prepend the worktree's `src` to `PYTHONPATH`.
The evaluation-only replay uses the exact production model loader and scorer:

```bash
python -m custom.mission_4.inference_replay \
  --case-dir /path/to/saved/rooms-0 --case-dir /path/to/saved/rooms-1 \
  --output /tmp/new-production-replay.json
python -m unittest custom.mission_4.test_mission_4 -v
python custom/mission_4/evaluate.py --run --source-root "$PWD" --output /tmp/new-m4-run
```

Read the team GPU/process status first; the simulator wrapper acquires its lock
once. Never run in, switch branches in, or reset another user's checkout.


## Final live standalone batch

| Case | Full mission | Seconds | Independent labels | Joint labels | Model load + inference (s) |
|---|---|---:|---:|---:|---:|
| rooms-0 | PASS | 217.26 | 2/3 | 3/3 | 59.39 |
| rooms-2 | PASS | 199.41 | 2/3 | 3/3 | 59.71 |
| rooms-3 | PASS | 166.76 | 3/3 | 3/3 | 60.46 |

All three official deliveries passed, all nine joint labels were correct, and
7/9 independent labels were correct. No pose-recovery event was needed in these
three runs. Runtime and shared-source hashes match the final local files.

## Continuous runs and control comparison

| Starting mission | M2 official result | M4 full result | M4 seconds | Failure stage |
|---|---|---|---:|---|
| Green cone, final Mission 2 | PASS | PASS | 166.69 | None; 3/3 room labels |
| Right-side blue cone, final Mission 2 | PASS | FAIL | 161.72 | South observation route, before inference |
| Same blue case, unchanged-code repeat | PASS | FAIL | 73.94 | Middle observation; pose recovery exhausted |
| Same blue case, previous M2 final turn (diagnostic control) | PASS, with controller stop-check error | FAIL | 100.08 | First observation route, before inference |

The main final configuration therefore passed **4/6 full Mission 4 attempts**:
3/3 standalone and 1/3 continuous. The separate previous-M2-ending comparison
also failed (not included in that 4/6). Across the four completed surveys,
12/12 joint labels and 9/12 independent labels were correct. The two failed
final-configuration runs produced no VLM predictions; the referee's 0/3 there
means no submitted answer, not three misclassified images.

In the first blue failure, the freshness wait recovered after 2.2109s, but
navigation later stalled around estimated (17.1,4.9); the final official pelvis
was (13.18,5.06). The repeat did not recover fresh localization within 8s and
ended at (11.52,10.81). The previous-M2-ending comparison timed out at the first
observation route, with final pelvis (14.23,21.16). Its M2 official PASS also
had a controller `Final rotation moved outside the estimated cone target`
error. The only differing custom Python file in that comparison is the Mission
2 controller; Mission 4 and all shared code are identical. The route failure
exists without the new stopping behavior, though these few trials cannot prove
equal failure probabilities between the two endings.

Even the passing green sequence ended at x=18.422, only 0.172m inside the
classroom boundary, despite a requested entry target x=20.0. That PASS does not
establish reliable localization. Shared navigation/localization remain the
integration owner's task; no truth-based correction or shared edit was made.

## Evidence and submission state

The adjacent evidence directory includes both ordered performance comparisons,
the 18-image production replay, all rejected recognition/quantization trials,
CPU logs, official standalone/continuous records, the failed comparison and
source audit. Ninety distinct CPU checks passed (19 M2, 25 M4, 46 shared).
Actual versions match the unchanged requirements: Torch 2.14.1+cpu and
Transformers 4.57.6. Default SmolVLM-500M weights are reused read-only; no API,
model download, GPU model inference or dependency upgrade was introduced.

Runtime hashes match the exact local M4 `257965b` and M2 `38c4a8f` sources.
The comparison changes only the M2 controller. Recorded base commits for the
remote worktrees do not by themselves identify their source overlays.
Large images/logs remain on OMEN at the recorded paths and are backed up under
`output/ys-missions/optimization-20261124/` on the Mac. Small JSON evidence and
reproduction scripts are committed; large weights and image archives are not.

For the full main test selection, run `evaluate.py` with `--case rooms-0
--case rooms-2 --case rooms-3`, then use an isolated combined checkout for
`--case after-cone --case after-cone-blue`. The attached launch scripts show the
exact environment and worktree paths; adapt those paths for another checkout.
Use a fresh output directory and the team simulator lock. Both mission branches
remain separate from main, ready for integration review with these limitations.
