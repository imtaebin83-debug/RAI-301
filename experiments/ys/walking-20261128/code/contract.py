"""Constants extracted from the pinned, unmodified competition loader."""
import numpy as np
JOINTS = ('left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
DEFAULT_POS = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0], dtype=float)
KP = np.array([40.179, 99.098, 40.179, 99.098, 28.501, 28.501, 40.179, 99.098, 40.179, 99.098, 28.501, 28.501, 40.179, 28.501, 28.501, 14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778, 14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778], dtype=float)
KD = np.array([2.558, 6.309, 2.558, 6.309, 1.814, 1.814, 2.558, 6.309, 2.558, 6.309, 1.814, 1.814, 2.558, 1.814, 1.814, 0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068, 0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068], dtype=float)
ACTION_SCALE = np.array([0.548, 0.351, 0.548, 0.351, 0.439, 0.439, 0.548, 0.351, 0.548, 0.351, 0.439, 0.439, 0.548, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075], dtype=float)
NOMINAL_DT = 0.002
CONTROL_DECIMATION = 10
POLICY_DT = NOMINAL_DT * CONTROL_DECIMATION
GAIT_PERIOD = 0.6
NUM_OBS = 98
