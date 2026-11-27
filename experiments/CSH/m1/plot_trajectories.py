"""Plot estimates separately from sparse official checkpoint measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
output = root.parents[2] / 'src/custom/docs/experiments/CSH_m1_20261127'
output.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharex=True, sharey=True, constrained_layout=True)
summary = []
for ax, relative, label in zip(axes,
        ('candidate-v1-comparison/roundtrip', 'normalized-navigation/roundtrip'),
        ('ICP guards; per-axis velocity limits', 'ICP guards; vector velocity limit')):
    directory = root / relative
    rows = [json.loads(s) for s in (directory / 'localization.jsonl').read_text().splitlines()]
    xy = np.asarray([r['pose'][:2] if 'pose' in r else [r['transform'][0][3], r['transform'][1][3]]
                     for r in rows if r.get('valid')])
    checkpoints = json.loads((directory / 'checkpoints.json').read_text())
    actual = np.asarray([r['official_checkpoint']['pelvis'][:2] for r in checkpoints])
    ax.plot(xy[:, 0], xy[:, 1], color='#2168a8', linewidth=1.1, label='sensor estimate')
    ax.scatter(actual[:, 0], actual[:, 1], marker='x', color='#c43b32', s=42,
               linewidths=1.8, label='official stopped checkpoint', zorder=5)
    for wall in (13., 18.):
        ax.axvline(wall, color='#777777', linewidth=1)
    ax.axvline(16., color='#999999', linewidth=.8, linestyle='--', label='requested route')
    ax.set(title=label, xlabel='world x (m)', xlim=(12.7, 18.3), ylim=(2.5, 19.4))
    ax.set_aspect('equal')
    ax.grid(alpha=.18)
    summary.append({'case': relative, 'max_estimated_cross_track_m': float(np.max(np.abs(xy[:, 0] - 16.))),
                    'max_official_checkpoint_error_m': max(r['actual_goal_error'] for r in checkpoints),
                    'max_checkpoint_estimate_error_m': max(r['estimate_error'] for r in checkpoints)})
axes[0].set_ylabel('world y (m)')
axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.09), fontsize=7)
fig.suptitle('45 m round trip: estimates are not continuous ground truth', fontsize=11)
fig.savefig(output / 'trajectory.png', dpi=170)
(root / 'trajectory_comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
