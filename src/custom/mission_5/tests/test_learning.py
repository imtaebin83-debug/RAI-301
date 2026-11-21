"""Synthetic pipeline checks; these are not demonstrations or robot-success tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

from custom.mission_5.bc.data import EpisodeDataset, read_manifest, state_stats, verify_dataset
from custom.mission_5.bc.frames import base_action_to_world
from custom.mission_5.bc.model import bc_loss, SpatialSummary, rotvec_to_sixd
from custom.mission_5.bc.policy import Policy
from custom.mission_5.bc.prepare import audit_episode, prepare
from custom.mission_5.bc.export import export_checkpoint


def write_episode(path, seed, success=1):
    rng = np.random.default_rng(seed)
    n = 5
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="val861-libero-panda-hdf5-v4", success=success, fps=10.,
                       task_seed=seed, state_frame="panda_base", action_frame="panda_base")
        f["time"] = np.arange(n) / 10
        for key in ("observation/image", "observation/wrist_image"):
            f.create_dataset(key, data=rng.integers(0, 255, (n, 256, 256, 3), dtype=np.uint8), compression="lzf")
        f["observation/state"] = rng.standard_normal((n, 8)).astype(np.float32)
        f["actions"] = rng.uniform(-1, 1, (n, 7)).astype(np.float32)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m5-pipeline-")
        self.root = Path(self.temp.name)
        for seed in range(5):
            write_episode(self.root / f"episode_{seed:06d}.h5", seed)

    def tearDown(self):
        self.temp.cleanup()

    def test_invalid_action_rejected(self):
        p = self.root / "episode_000000.h5"
        with h5py.File(p, "r+") as f:
            f["actions"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            audit_episode(p)

    def test_official_failure_cannot_be_a_success_label(self):
        p=self.root/'episode_000000.h5'
        with h5py.File(p,'r+') as f:
            f.attrs['official_success']=False
        with self.assertRaisesRegex(ValueError,'official'):
            audit_episode(p)

    def test_placement_id_keeps_repeated_placement_in_one_split(self):
        for i in (0,1):
            with h5py.File(self.root/f'episode_{i:06d}.h5','r+') as f:
                f.attrs['placement_id']='same-placement'
        report=prepare(self.root,self.root/'manifest.json')
        locations=[s for s,names in report['splits'].items()
                   if 'episode_000000.h5' in names or 'episode_000001.h5' in names]
        self.assertEqual(len(locations),1)

    def test_manifest_cannot_replace_actions_with_diagnostics(self):
        path=self.root/'manifest.json'
        report=prepare(self.root,path)
        for record in report['episodes']:
            record['action_key']='diagnostics/pencil_pose_world'
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'recorded normalized actions'):
            read_manifest(path)

    def test_changed_episode_is_rejected_before_training(self):
        path=self.root/'manifest.json'
        report=prepare(self.root,path)
        verify_dataset(path)
        with h5py.File(self.root/report['splits']['train'][0],'r+') as f:
            f['actions'][0,0]=.123456
        with self.assertRaisesRegex(ValueError,'changed after audit'):
            verify_dataset(path)

    def test_spatial_features_retain_horizontal_position(self):
        left=torch.zeros(1,1,5,5); right=left.clone()
        left[0,0,2,0]=4.; right[0,0,2,4]=4.
        a,b=SpatialSummary()(left),SpatialSummary()(right)
        self.assertLess(float(a[0,0]),-.9)
        self.assertGreater(float(b[0,0]),.9)
        self.assertAlmostEqual(float(a[0,-1]),float(b[0,-1]))

    def test_rotvec_branch_cut_has_continuous_proprio_features(self):
        a=rotvec_to_sixd(torch.tensor([[torch.pi,0.,0.],[0.,0.,0.]]))
        b=rotvec_to_sixd(torch.tensor([[-torch.pi,0.,0.],[0.,0.,0.]]))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
        self.assertTrue(torch.isfinite(a).all())

    def test_in_progress_recording_is_not_an_audited_episode(self):
        (self.root/'episode_999.recording.h5').write_bytes(b'incomplete')
        report=prepare(self.root,self.root/'manifest.json')
        self.assertTrue(report['ready'])
        self.assertEqual(len(report['episodes']),5)

    def test_groups_do_not_leak_and_normalization_uses_train_only(self):
        write_episode(self.root / "episode_000099.h5", 0)
        manifest = self.root / "manifest.json"
        report = prepare(self.root, manifest)
        self.assertTrue(report["ready"])
        splits = report["splits"]
        same_seed = [split for split, files in splits.items()
                     if "episode_000000.h5" in files or "episode_000099.h5" in files]
        self.assertEqual(len(same_seed), 1)
        mean, _ = state_stats(manifest)
        val_file = self.root / splits["val"][0]
        with h5py.File(val_file, "r+") as f:
            f["observation/state"][:] = 1e6
        other_mean, _ = state_stats(manifest)
        np.testing.assert_array_equal(mean, other_mean)
        loaded = json.loads(manifest.read_text())
        loaded["policy_inputs"].append("diagnostics/pencil_pose_world")
        manifest.write_text(json.dumps(loaded))
        with self.assertRaisesRegex(ValueError, "Only"):
            read_manifest(manifest)

    def test_horizon_padding_is_masked(self):
        manifest = self.root / "manifest.json"
        prepare(self.root, manifest)
        mean, std = state_stats(manifest)
        dataset = EpisodeDataset(manifest, "train", 8, mean, std)
        images, state, actions, mask = dataset[4]
        self.assertEqual(images.shape, (2, 3, 256, 256))
        self.assertEqual(int(mask.sum()), 1)
        prediction = torch.zeros(1, 8, 7)
        a = bc_loss(prediction, actions[None], mask[None])
        actions[1:] = 1e4
        b = bc_loss(prediction, actions[None], mask[None])
        self.assertEqual(float(a), float(b))

    def test_camera_pose_changes_are_not_silently_mixed(self):
        for i, path in enumerate(sorted(self.root.glob("episode_*.h5"))):
            with h5py.File(path, "r+") as f:
                f.attrs["agentview_pos_local_xyz"] = np.array([float(i), 0., 0.])
        report = prepare(self.root, self.root / "manifest.json")
        self.assertFalse(report["ready"])

    def test_base_to_world_rotation(self):
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        action = np.array([.2, 0., 0., .3, 0., 0., -1.])
        np.testing.assert_allclose(base_action_to_world(action, rotation), [0., -.2, 0., 0., -.3, 0., -1.], atol=1e-7)
        with self.assertRaises(ValueError):
            base_action_to_world(action, np.zeros((3, 3)))

    def test_train_resume_and_policy_ignores_privileged_fields(self):
        manifest = self.root / "manifest.json"
        prepare(self.root, manifest)
        run = self.root / "run"
        command = [sys.executable, "-m", "custom.mission_5.bc.train", "--manifest", str(manifest), "--out", str(run),
                   "--device", "cpu", "--workers", "0", "--batch-size", "2", "--horizon", "2",
                   "--image-size", "32", "--max-batches", "1"]
        subprocess.run(command + ["--epochs", "1"], check=True, capture_output=True, text=True)
        subprocess.run(command + ["--epochs", "2", "--resume", str(run / "last.pt")],
                       check=True, capture_output=True, text=True)
        saved = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(saved["epoch"], 2)
        self.assertIsNone(saved["official_task_success"])
        policy = Policy(run / "best.pt", "cpu")
        observation = {"image": np.zeros((256, 256, 3), np.uint8),
                       "wrist_image": np.zeros((256, 256, 3), np.uint8), "state": np.zeros(8, np.float32)}
        a = policy.act(observation)["actions"]
        observation["pencil_pose_world"] = np.full(7, np.nan)
        observation["holder_pose_world"] = np.full(7, 1e12)
        b = policy.act(observation)["actions"]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (2, 7))
        self.assertTrue(np.isfinite(a).all() and (np.abs(a) <= 1).all())
        compact=run/'export.pt'
        export_checkpoint(run/'best.pt',compact)
        np.testing.assert_array_equal(a,Policy(compact,'cpu').act(observation)['actions'])
        exported=torch.load(compact,weights_only=True,map_location='cpu')
        self.assertNotIn('optimizer',exported)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
