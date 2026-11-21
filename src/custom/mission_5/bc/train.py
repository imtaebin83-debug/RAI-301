"""Train the two-camera BC baseline; validation loss is not task success."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeDataset, verify_dataset, state_stats
from .model import TwoCameraBC, bc_loss


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-batches", type=int, help="For pipeline tests only; recorded in checkpoints")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.horizon, args.image_size, args.threads) <= 0 or args.workers < 0:
        p.error("sizes/epochs/threads must be positive; workers must be nonnegative")
    if args.max_batches is not None and args.max_batches <= 0:
        p.error("max-batches must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu for pipeline validation")
    manifest, _, by_name = verify_dataset(args.manifest)
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config = {"horizon": args.horizon, "image_size": args.image_size}
    mean, std = state_stats(args.manifest)
    generator = torch.Generator().manual_seed(args.seed)
    model = TwoCameraBC(**config).to(device)
    model.set_state_normalization(mean,std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch0, best = 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (checkpoint.get('format')!='m5-two-camera-bc-v3' or
                checkpoint["manifest_sha256"] != fingerprint or checkpoint["config"] != config):
            raise ValueError("Resume requires the identical manifest and model configuration")
        if args.out.resolve() != args.resume.resolve().parent:
            raise ValueError("Resume --out must be the checkpoint's run directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if 'python_rng' in checkpoint:
            random.setstate(checkpoint['python_rng'])
        generator.set_state(checkpoint["loader_rng"])
        if amp and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        epoch0, best = checkpoint["epoch"], checkpoint["best_val"]
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaders = {split: DataLoader(EpisodeDataset(args.manifest, split, args.horizon, mean, std),
                                batch_size=args.batch_size, shuffle=split == "train",
                                num_workers=args.workers, generator=generator if split == "train" else None,
                                pin_memory=amp) for split in ("train", "val")}
    print(json.dumps({"device": str(device), "frames": {s: len(x.dataset) for s, x in loaders.items()},
                      "config": config, "max_batches": args.max_batches}), flush=True)
    for epoch in range(epoch0 + 1, args.epochs + 1):
        t0 = time.monotonic()
        metrics = {}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            total, count = 0., 0
            for batch_index, (images, state, actions, mask) in enumerate(loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                images, state, actions, mask = [x.to(device, non_blocking=amp) for x in (images, state, actions, mask)]
                with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
                    loss = bc_loss(model(images, state), actions, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite BC loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                valid = int(mask.sum())
                total += float(loss.detach()) * valid
                count += valid
            if count == 0:
                raise ValueError("No valid action targets")
            metrics[split] = total / count
        improved = metrics["val"] < best
        best = min(best, metrics["val"])
        fps = float(np.median([by_name[n]["fps"] for n in manifest["splits"]["train"]]))
        checkpoint = {"format": "m5-two-camera-bc-v3", "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "config": config, "state_mean": torch.from_numpy(mean), "state_std": torch.from_numpy(std),
                      "epoch": epoch, "best_val": best, "manifest_sha256": fingerprint,
                      "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                      "python_rng": random.getstate(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if amp else [],
                      "fps": fps, "action_frame": "panda_base", "max_batches": args.max_batches,
                      "input_keys": manifest["policy_inputs"], "official_task_success": None,
                      "collection_camera_metadata": by_name[manifest["splits"]["train"][0]]["camera_metadata"]}
        save_checkpoint(args.out / "last.pt", checkpoint)
        if improved:
            save_checkpoint(args.out / "best.pt", checkpoint)
        row = {"epoch": epoch, "train_loss": metrics["train"], "val_loss": metrics["val"],
               "seconds": time.monotonic() - t0, "official_task_success": None}
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
