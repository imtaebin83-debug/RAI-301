"""Base-to-world action conversion for the provided normalized ROS controller."""
import numpy as np


def base_action_to_world(action, world_from_base_rotation):
    action = np.asarray(action, dtype=np.float64)
    rotation = np.asarray(world_from_base_rotation, dtype=np.float64)
    if action.shape != (7,) or rotation.shape != (3, 3):
        raise ValueError("Expected action[7], rotation[3,3]")
    if not np.isfinite(action).all() or not np.isfinite(rotation).all():
        raise ValueError("Nonfinite action or rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ValueError("Expected a proper rotation matrix")
    result = action.copy()
    result[:3] = rotation @ action[:3]
    result[3:6] = rotation @ action[3:6]
    # Preserve direction if rotated normalized components exceed the ROS cube bound.
    # One shared factor preserves the desired 6-D twist direction; no per-axis clipping.
    result[:6] /= max(1., float(np.abs(result[:6]).max()))
    result[6] = -1. if action[6] < 0 else 1.
    return result.astype(np.float32)
