"""Original referee measurements at stops; no continuous ground-truth claim."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
output = root.parents[2] / 'src/custom/docs/experiments/CSH_m1_20261127/v2'
output.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
measurements = []
for offset, directory, label, color in (
        (-.19, 'v2-baseline-roundtrip/roundtrip', 'v2 main baseline', '#bb632c'),
        (.19, 'v2-navigation/roundtrip', 'guarded ICP + gyro + vector control', '#2168a8')):
    rows = json.loads((root / directory / 'checkpoints.json').read_text())
    assert len(rows) == 4
    for ax, key in zip(axes, ('actual_goal_error', 'estimate_error')):
        values = [row[key] for row in rows]
        bars = ax.bar(np.arange(4)+offset, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=7)
        ax.set(xticks=np.arange(4), xticklabels=('north 1', 'south', 'north 2', 'finish'),
               ylabel='error (m)', ylim=(0, max(.42, max(values)*1.25)))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    measurements.append({'case': directory, 'checkpoints': rows})
axes[0].set_title('Actual stopped position to requested goal')
axes[1].set_title('Sensor estimate to actual stopped position')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False, fontsize=8)
fig.suptitle('Starter v2: one 45 m round trip each; four stopped checkpoints', fontsize=11)
fig.savefig(output / 'checkpoint_errors.png', dpi=170)
(output / 'checkpoint_errors.json').write_text(json.dumps(measurements, indent=2)+'\n')
