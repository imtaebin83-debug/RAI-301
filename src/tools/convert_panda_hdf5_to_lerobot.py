"""Convert collect_panda.py episodes (episode_*.h5) to a LeRobot dataset for OpenPI.

Run with OpenPI's Python (src/openpi, after `uv sync`):

    cd src/openpi
    uv run ../tools/convert_panda_hdf5_to_lerobot.py ../custom/mission_5/dataset --repo-id local/mission5_panda

Only episodes saved as success are used (--include-failed takes all).  Several episode
folders may be given; episodes are renumbered in the order given.  The dataset is written
to $HF_LEROBOT_HOME/<repo-id> (default ~/.cache/huggingface/lerobot/<repo-id>) with the
feature names image, wrist_image, state, actions (as OpenPI's LeRobotLiberoDataConfig reads).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

SUPPORTED_SCHEMAS = {
    "val861-libero-panda-hdf5-v2",
    "val861-libero-panda-hdf5-v3",
    "val861-libero-panda-hdf5-v4",
}
# Episodes that disagree on any of these were not recorded on the same task contract
# (camera moved, other control rate or action scale) and must not share one dataset.
CONTRACT_ATTRS = (
    "schema", "fps", "state_frame", "action_frame",
    "action_position_scale_m", "action_rotation_scale_rad",
    "agentview_pos_local_xyz", "agentview_quat_local_wxyz", "agentview_fovy_deg",
    "wrist_fovy_deg",
)


def _episodes(roots: list[Path], include_failed: bool) -> list[Path]:
    result = []
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"raw dataset folder not found: {root}")
        kept = 0
        paths = sorted(root.glob("episode_*.h5"))
        for path in paths:
            with h5py.File(path, "r") as episode:
                schema = str(episode.attrs.get("schema", ""))
                if schema not in SUPPORTED_SCHEMAS:
                    raise SystemExit(f"{path}: unsupported schema {schema!r}")
                if include_failed or int(episode.attrs.get("success", -1)) == 1:
                    result.append(path)
                    kept += 1
        print(f"{root}: {kept} of {len(paths)} episodes used")
    return result


def _check_contract(paths: list[Path]) -> None:
    reference: dict[str, object] = {}
    for path in paths:
        with h5py.File(path, "r") as episode:
            for name in CONTRACT_ATTRS:
                if name not in episode.attrs:
                    continue
                value = episode.attrs[name]
                if name not in reference:
                    reference[name] = (value, path)
                    continue
                expected, source = reference[name]
                same = (
                    str(value) == str(expected)
                    if isinstance(value, (str, bytes))
                    else np.allclose(np.asarray(value, float), np.asarray(expected, float),
                                     rtol=0.0, atol=1e-6)
                )
                if not same:
                    raise SystemExit(
                        f"{path}: {name}={value!r} differs from {expected!r} in {source}.\n"
                        "These episodes were recorded with another camera/control setup; "
                        "export them as a separate dataset."
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dirs", type=Path, nargs="+")
    parser.add_argument("--repo-id", default="local/mission5_panda",
                        help="LeRobot repository id and local directory name")
    parser.add_argument("--include-failed", action="store_true",
                        help="include failed/unknown episodes (normally wrong for behavior cloning)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing LeRobot dataset with the same repo id")
    args = parser.parse_args()

    paths = _episodes([p.expanduser().resolve() for p in args.raw_dirs], args.include_failed)
    if not paths:
        raise SystemExit("no " + ("" if args.include_failed else "successful ") + "episodes found")
    _check_contract(paths)

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_path} already exists; rerun with --overwrite to replace it")
        shutil.rmtree(output_path)

    with h5py.File(paths[0], "r") as first:
        fps = int(round(float(first.attrs["fps"])))
    image_feature = {"dtype": "image", "shape": (256, 256, 3),
                     "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": image_feature,
            "wrist_image": image_feature,
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    total_frames = 0
    for path in paths:
        with h5py.File(path, "r") as episode:
            image = episode["observation/image"]
            wrist = episode["observation/wrist_image"]
            state = episode["observation/state"]
            actions = episode["actions" if "actions" in episode else "action"]
            lengths = {len(image), len(wrist), len(state), len(actions)}
            if len(lengths) != 1:
                raise SystemExit(f"{path}: inconsistent frame counts {lengths}")
            if image.shape[1:] != (256, 256, 3) or wrist.shape[1:] != (256, 256, 3):
                raise SystemExit(f"{path}: cameras must be uint8 [T,256,256,3]")
            if state.shape[1:] != (8,) or actions.shape[1:] != (7,):
                raise SystemExit(f"{path}: expected state 8-D and actions 7-D")
            task = str(episode.attrs["task_instruction"])
            for index in range(len(image)):
                dataset.add_frame(
                    {
                        "image": np.asarray(image[index], dtype=np.uint8),
                        "wrist_image": np.asarray(wrist[index], dtype=np.uint8),
                        "state": np.asarray(state[index], dtype=np.float32),
                        "actions": np.asarray(actions[index], dtype=np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()
            total_frames += len(image)
            print(f"converted {path.parent.name}/{path.name}: {len(image)} frames")

    print(f"LeRobot dataset: {output_path}")
    print(f"episodes={len(paths)} frames={total_frames} fps={fps} repo_id={args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
