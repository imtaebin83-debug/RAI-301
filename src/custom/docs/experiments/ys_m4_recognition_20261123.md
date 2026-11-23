# Mission 4 recognition experiments — ys — November 23, 2026

Runtime recognition remains at tested commit `6c17eec`. Neither experimental
scorer is suitable for deployment. Shared code was not changed. Latest main
was checked at `7542e7e`; its changes concern Mission 5, not shared localization.

## Findings

The original model generated " Medical.", " Classroom.", " Medical." on the
previously ambiguous medical/cafeteria/classroom survey. Thus generating text
alone did not solve recognition. Scoring uniformly formatted answers with a
leading space, title case and period produced a confidently wrong joint
assignment medical/classroom/cafeteria (margin 1.441), while the original
scorer rejected ambiguity. The formatted variant was rejected and stopped.

Subtracting scores from a neutral gray 640x480 image improved individual
predictions only from 6/18 to 8/18 on the six saved final-regression surveys.
These cases were already inspected and are not new held-out data. A fixed
per-label score offset leaves the one-of-each joint assignment and its margin
unchanged algebraically: every permutation subtracts the same sum. It therefore
does not improve delivery decisions and adds about one image inference per
worker. It was not deployed. The previous actual-image crop retry remains.

## Deliverable and verification

`custom/mission_4/recognition_replay.py` is evaluation only. It requires explicit
saved case directories and a new output path, loads existing local model
weights on CPU with four Torch threads, and records image hashes, true labels,
predictions and joint assignment. Only this evaluator reads official records;
the robot controller never imports it. It supports the two rejected variants
so the findings can be reproduced instead of retuning against anecdotes.

21 existing Mission 4 CPU checks passed. Both evaluator modes completed real CPU-model runs successfully; their
JSON outputs and process log are attached. The formatted run reproduced the
wrong joint assignment, and the neutral run retained the original joint order. No new model, package, shared environment setting,
mission controller or common localization/navigation source was changed.

Example in the verified environment with the ys source on PYTHONPATH:

```bash
python -m custom.mission_4.recognition_replay --variant formatted \
  --case-dir /tmp/ys-m4-candidate-reattempt-recovery/rooms-3 \
  --output /tmp/new-formatted-replay.json
```

This work establishes rejected hypotheses and a reusable evaluation tool,
not a claimed improvement in deployed recognition accuracy or delivery rate.
