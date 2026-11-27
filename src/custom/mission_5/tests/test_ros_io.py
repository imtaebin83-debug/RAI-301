import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from custom.mission_5.ros_io import (ARM_NAMES, FINGER_NAMES, PandaKinematics, PandaROSIO,
        base_action_to_world, joint_mapping, rgb_array, public_base_axes)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mission_5.ros_io import (ARM_NAMES, FINGER_NAMES, PandaKinematics, PandaROSIO,
        base_action_to_world, joint_mapping, rgb_array, public_base_axes)


class ActionContractTests(unittest.TestCase):
    def test_public_axes_have_fixed_yaw_and_no_world_translation(self):
        axes = public_base_axes()
        np.testing.assert_allclose(axes[:3, 3], 0.)
        np.testing.assert_allclose(axes[:3, :3], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(axes[3], [0, 0, 0, 1])

    def test_base_yaw_rotates_translation_and_rotvec(self):
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        action = base_action_to_world([.2, -.4, .1, .3, .1, -.2, -1], world)
        np.testing.assert_allclose(action, [-.4, -.2, .1, .1, -.3, -.2, -1], atol=1e-8)

    def test_rotation_is_left_multiplied_in_equivalent_frames(self):
        rw = Rotation.from_euler('xyz', [.1, -.2, .5]).as_matrix()
        rb = Rotation.from_euler('xyz', [2., .1, -.7]).as_matrix()
        action = np.array([.1, .2, -.1, .05, -.02, .1, 1.])
        world = np.eye(4); world[:3, :3] = rw
        command = base_action_to_world(action, world)
        expected = rw @ Rotation.from_rotvec(action[3:6] * .5).as_matrix() @ rb
        actual = Rotation.from_rotvec(command[3:6] * .5).as_matrix() @ (rw @ rb)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_diagonal_limits_preserve_direction(self):
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler('z', math.pi / 4).as_matrix()
        result = base_action_to_world([1, 1, 0, 0, 0, 0, 1], world)
        self.assertLessEqual(float(np.abs(result).max()), 1)
        np.testing.assert_allclose(result[:3], [0, 1, 0], atol=1e-8)

    def test_invalid_action_cannot_reach_command(self):
        for action in ([0]*6, [0]*6+[float('nan')]):
            with self.assertRaises(ValueError):
                base_action_to_world(action, np.eye(4))
        bad = np.eye(4); bad[0,0] = 2
        with self.assertRaises(ValueError):
            base_action_to_world([0]*7, bad)


class SensorContractTests(unittest.TestCase):
    def test_rgb_padding_and_bgr_order(self):
        image = np.zeros((256, 256*3+8), dtype=np.uint8)
        image[:, :256*3] = np.tile([9, 22, 35], (256, 256))
        message = SimpleNamespace(encoding='bgr8', width=256, height=256,
                                  step=image.shape[1], data=image.tobytes())
        decoded = rgb_array(message)
        np.testing.assert_array_equal(decoded[20, 23], [35, 22, 9])
        self.assertTrue(decoded.flags.c_contiguous)

    def test_joint_state_names_accept_namespace_and_reject_missing(self):
        names = ARM_NAMES + FINGER_NAMES
        message = SimpleNamespace(name=['panda/'+n for n in names], position=list(range(9)))
        values = joint_mapping(message)
        self.assertEqual(values[FINGER_NAMES[1]], 8)
        message.name = message.name[:-1]
        message.position = message.position[:-1]
        with self.assertRaises(ValueError):
            joint_mapping(message)

    def test_camera_shape_is_not_silently_resized(self):
        message = SimpleNamespace(encoding='rgb8',width=2,height=2,step=6,data=bytes(12))
        with self.assertRaises(ValueError):
            rgb_array(message)


class KinematicContractTests(unittest.TestCase):
    def test_fk_uses_only_robot_description_and_joint_sensors(self):
        try:
            fk = PandaKinematics()
        except (FileNotFoundError, ValueError) as error:
            self.skipTest('Full competition robot model unavailable: '+str(error))
        home = [0., -.161037389, 0., -2.44459747, 0., 2.22675220, math.pi/4]
        joints = dict(zip(ARM_NAMES, home))
        joints.update({FINGER_NAMES[0]:.02,FINGER_NAMES[1]:-.02})
        state = fk.state(joints)
        self.assertEqual(state.shape, (8,))
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_allclose(state[-2:], [.02,-.02])
        np.testing.assert_allclose(fk.base_from_eef[:3,:3].T @ fk.base_from_eef[:3,:3],np.eye(3),atol=1e-6)


class V2PandaBoundaryTests(unittest.TestCase):
    def test_two_rgb_and_fk_work_without_any_world_tf(self):
        stamp = SimpleNamespace(sec=2, nanosec=0)
        header = SimpleNamespace(stamp=stamp)
        rgb = SimpleNamespace(header=header, encoding='rgb8', width=256, height=256,
                              step=256*3, data=bytes(256*256*3))
        names = ARM_NAMES + FINGER_NAMES
        joints = SimpleNamespace(header=header, name=names, position=[0.]*7+[.02, -.02])
        messages = {'/panda/agentview/image_raw': rgb,
                    '/panda/wrist_camera/image_raw': rgb, '/panda/joint_states': joints}
        subscriptions, publications = [], {}

        def latest(topic, kind, timeout=0.):
            subscriptions.append(topic)
            return messages.get(topic)

        def publish_to(kind, topic, qos):
            publications[topic] = []
            return SimpleNamespace(publish=publications[topic].append)

        def forbidden_tf():
            self.fail('Panda world/map TF must not be requested by the v2 adapter')

        mission = SimpleNamespace(latest=latest, create_publisher=publish_to, _tf=forbidden_tf)
        expected_state = np.array([.4, .1, .2, 0., 0., 0., .02, -.02], dtype=np.float32)
        fk = SimpleNamespace(state=lambda mapping: expected_state.copy())
        io = PandaROSIO(mission, kinematics=fk)
        observation = io.observation(timeout=.1)
        self.assertEqual(set(observation), {'time', 'image', 'wrist_image', 'state'})
        self.assertEqual(observation['image'].shape, (256, 256, 3))
        self.assertEqual(observation['wrist_image'].shape, (256, 256, 3))
        np.testing.assert_allclose(observation['state'], expected_state)
        self.assertEqual(set(subscriptions), set(messages))
        command = io.command([.2, -.4, .1, .3, .1, -.2, -1])
        np.testing.assert_allclose(command, [-.4, -.2, .1, .1, -.3, -.2, -1], atol=1e-8)
        self.assertEqual(len(publications['/panda/eef_delta_command']), 1)
        first = io.robot_base()
        first[:3, 3] = 999.
        np.testing.assert_allclose(io.robot_base()[:3, 3], 0.)


if __name__=='__main__':
    unittest.main()
