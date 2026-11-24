# Mission 2 continuous-operation follow-up — ys — November 24, 2026

Mission 2 runtime remains the verified `ed2ce50` fix. Main remains `7542e7e`;
no new shared localization/navigation update was available. Shared sources,
settings and environments were not changed.

## Previous verified position

The November 23 final candidate passed 15/15 standalone cases and 2/3 continuous
Mission 1→2 cases. No final-candidate detection or stop-box failure occurred.
The one failure was a shared route stall before detection. See
`ys_m2_fixes_20261123.md`; the earlier baseline failures are retained separately.

## Latest Mission 2 → Mission 4 tests

The isolated checkout `/home/aril/rai-worktrees/ys-m2-m4-20261124` combines
published Mission 2 `ed2ce50` with published Mission 4 `4972de3`. These are
continuous runs from the stock spawn, with no reset between missions, original
simulator/referee, map correction OFF and the team GPU lock.

| Run | Mission 2 | Official absolute x / y stop error | Mission 4 |
|---|---|---|---|
| latest-combined-1124 | PASS 49.35s | 0.045 / 0.004m | PASS 191.18s; all rooms correct and classroom delivery complete |
| latest-combined-repeat-1124 | PASS 50.11s | 0.033 / 0.006m | FAIL 163.18s; stalled en route to the last survey point |

**Mission 2 passed 2/2; complete Mission 2→4 passed 1/2.** This demonstrates
continuous completion is possible with the current pair, but it remains
intermittent. Do not call the shared blocker solved. The failed repeat never
reached the last room observation or VLM inference and cannot be attributed to
the new recognition experiment, which was not running in its controller.

The combined code passed 37 mission CPU tests and 46 shared tests (83 total).
Every record includes the actual imported custom-source hashes. The combined
checkout's base commit alone does not identify its Mission 4 overlay.

With the separately tested Mission 4 description scorer (`21a6fd3`), another
continuous run passed both missions: **Mission 2 PASS (48.84s); Mission 4
PASS (223.95s)**, all rooms correct and delivery complete. The combined source
hashes match Mission 2 `ed2ce50` plus the Mission 4 overlay, with no Mission 2
runtime change.
The failed earlier Mission 4 run remains part of the record.

The final Mission 4 heading-budget revision (`d768848`) also passed continuously:
**Mission 2 PASS (49.37s); Mission 4 PASS (183.80s)**, with correct room labels
and delivery. Mission 2's official absolute x/y stop errors were 0.047/0.023m.
All source hashes match the stated Mission 2 and Mission 4 revisions.

**Today Mission 2 passed 4/4 continuous attempts**, all with the same green-cone
layout. The four complete Mission 2→4 sequences passed 3/4 across the three
Mission 4 revisions; this mixed-version total is descriptive, not a final-code
reliability estimate. The latest pair completed its one continuous check.
No new Mission 2 detection or stop-position failure was observed. Broader colour
coverage remains the November 23 matrix, not four new all-colour tests.

## Reproduction and ownership

Use the existing `haic_rai301_verify` environment, prepend the combined `src`
to `PYTHONPATH`, confirm the GPU is available, then run from that `src`:

```bash
python custom/mission_4/evaluate.py --run --case after-cone \
  --source-root "$PWD" --output /tmp/ys-new-combined-check
```

Use a fresh output path. The Mission 2 branch alone does not contain the separately
owned latest Mission 4 code. Compact official records accompany this report;
full logs and camera captures remain in `output/ys-missions/continuation-20261124/`
on the Mac and the corresponding `/tmp/ys-m4-candidate-*1124/` folders on OMEN.

No further Mission 2 runtime change is justified by these successful
approaches. The integration owner still needs to investigate pose drift and
route progress through the corridor and room entrances, then rerun Mission
1→2 and Mission 2→4 with both personal branches' tested code.
