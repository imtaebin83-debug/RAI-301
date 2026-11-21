from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare import INPUT_KEYS


def read_manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if not m.get("ready"):
        raise ValueError("Manifest is not ready; audit successful demonstrations first")
    if tuple(m.get("policy_inputs", [])) != INPUT_KEYS:
        raise ValueError("Only the two camera images and robot proprioception are allowed as policy inputs")
    root = Path(m["root"]).resolve()
    by_name = {r["file"]: r for r in m["episodes"]}
    if len(by_name)!=len(m['episodes']):
        raise ValueError("Duplicate episode paths in manifest")
    seen = set()
    group_sets = []
    for split in ("train", "val", "test"):
        names = m["splits"][split]
        if not names or len(names)!=len(set(names)) or seen.intersection(names):
            raise ValueError("Each split needs distinct nonempty episode lists")
        seen.update(names)
        if any(n not in by_name for n in names):
            raise ValueError("Split references unknown episode")
        group_sets.append({by_name[n]["group"] for n in names})
        for name in names:
            if not (root / name).resolve().is_relative_to(root) or by_name[name]["success"] != 1:
                raise ValueError("Unsafe path or non-success episode in manifest")
            if by_name[name].get('action_key') not in ('action','actions'):
                raise ValueError('Only recorded normalized actions can be training targets')
    if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1:]):
        raise ValueError("Initial-condition leakage between splits")
    return m, root, by_name


def state_stats(manifest):
    m, root, _ = read_manifest(manifest)
    count, total, square = 0, np.zeros(8), np.zeros(8)
    for name in m["splits"]["train"]:
        with h5py.File(root / name, "r") as f:
            x = np.asarray(f["observation/state"][:], dtype=np.float64)
        count += len(x)
        total += x.sum(0)
        square += (x * x).sum(0)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def verify_dataset(manifest):
    """Detect changes after the audit, before any training/validation is run."""
    m,root,records=read_manifest(manifest)
    for name in (name for split in ('train','val','test') for name in m['splits'][split]):
        expected=records[name].get('sha256')
        if not expected:
            raise ValueError('Episode hash missing; create a new audited manifest')
        digest=hashlib.sha256()
        with (root/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest()!=expected:
            raise ValueError('Episode changed after audit: '+name)
    return m,root,records


class EpisodeDataset(Dataset):
    def __init__(self, manifest, split, horizon, mean, std):
        m, self.root, by_name = read_manifest(manifest)
        self.records = [by_name[n] for n in m["splits"][split]]
        self.ends = np.cumsum([r["frames"] for r in self.records]).tolist()
        self.horizon = horizon
        if int(horizon)<1:
            raise ValueError('horizon must be positive')
        self.mean, self.std = np.asarray(mean), np.asarray(std)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode = bisect.bisect_right(self.ends, index)
        start = index - (0 if episode == 0 else self.ends[episode - 1])
        r = self.records[episode]
        stop = min(start + self.horizon, r["frames"])
        with h5py.File(self.root / r["file"], "r") as f:
            # Deliberate allowlist: diagnostics are never read here.
            images = np.stack([f[k][start] for k in INPUT_KEYS[:2]])
            state = (np.asarray(f[INPUT_KEYS[2]][start], np.float32) - self.mean) / self.std
            actions = np.asarray(f[r["action_key"]][start:stop], np.float32)
        valid = len(actions)
        actions = np.pad(actions, ((0, self.horizon - valid), (0, 0)), mode="edge")
        mask = np.arange(self.horizon) < valid
        return (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255,
                torch.from_numpy(state.astype(np.float32)), torch.from_numpy(actions),
                torch.from_numpy(mask))
