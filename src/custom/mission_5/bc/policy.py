"""Inference adapter for the supplied class PandaLearningRuntime."""
import os
from pathlib import Path

import numpy as np
import torch

from .model import TwoCameraBC, decode_actions


class Policy:
    def __init__(self, checkpoint, device="cpu"):
        torch.set_num_threads(max(1,min(8,int(os.environ.get('M5_TORCH_THREADS','2')))))
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "m5-two-camera-bc-v3" or saved["action_frame"] != "panda_base":
            raise ValueError("Incompatible policy checkpoint")
        if tuple(saved.get('input_keys',[])) != ('observation/image','observation/wrist_image','observation/state'):
            raise ValueError('Checkpoint declares non-whitelisted policy inputs')
        self.model = TwoCameraBC(**saved["config"]).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.mean = saved["state_mean"].numpy()
        self.std = saved["state_std"].numpy()
        self.fps = saved["fps"]
        if (self.mean.shape!=(8,) or self.std.shape!=(8,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std<=0) or not 0<float(self.fps)<=100):
            raise ValueError('Invalid checkpoint normalization or control rate')
        if (not np.allclose(self.model.input_mean.cpu().numpy(),self.mean) or
                not np.allclose(self.model.input_std.cpu().numpy(),self.std)):
            raise ValueError('Checkpoint model and preprocessing normalization disagree')
        self.manifest_sha256=saved['manifest_sha256']

    def reset(self):
        pass

    @torch.inference_mode()
    def act(self, observation):
        # Additional fields such as pencil_pose_world are deliberately ignored.
        frames = [np.asarray(observation[k]) for k in ("image", "wrist_image")]
        if any(x.shape != (256, 256, 3) or x.dtype != np.uint8 for x in frames):
            raise ValueError("Expected two 256x256 uint8 RGB frames, without preview-only rotations")
        state = np.asarray(observation["state"], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 8-D Panda-base proprioception")
        images = torch.from_numpy(np.stack(frames).copy()).permute(0, 3, 1, 2).float()[None].to(self.device) / 255
        state = torch.from_numpy((state - self.mean) / self.std)[None].to(self.device)
        actions = decode_actions(self.model(images, state))[0].cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite policy output")
        return {"actions": actions}


def create_policy():
    path = os.environ.get("M5_CHECKPOINT")
    if not path:
        raise RuntimeError("Set M5_CHECKPOINT to the team's trained best.pt file")
    return Policy(Path(path), os.environ.get("M5_DEVICE", "cpu"))
