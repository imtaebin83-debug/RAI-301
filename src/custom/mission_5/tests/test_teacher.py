"""Numerical/vision regressions, not simulated task-success demonstrations."""
import math
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from custom.mission_5.vision_teacher import CameraModel, TeacherConfig, VisionTeacher, measure_scene


def image():
    rgb=np.zeros((256,256,3),np.uint8)
    rgb[:]=[160,95,25]
    cv2.rectangle(rgb,(55,96),(115,101),(255,195,15),-1)
    cv2.rectangle(rgb,(157,64),(196,113),(90,100,110),-1)
    return rgb


class TeacherGeometryTests(unittest.TestCase):
    def test_ray_plane_roundtrip(self):
        camera=CameraModel.panda_agentview_default()
        pixels=np.array([[80,98],[175,76]],float)
        points=camera.on_plane(pixels,.15)
        optical=(points-camera.base_from_optical[:3,3])@camera.base_from_optical[:3,:3]
        uvw=optical@camera.k.T
        np.testing.assert_allclose(uvw[:,:2]/uvw[:,2:],pixels,atol=1e-8)

    def test_invalid_ray_rejected(self):
        with self.assertRaises(ValueError):
            CameraModel(np.eye(3),np.eye(4)).on_plane([[0,0]],-1)

    def test_sensor_detection(self):
        m=measure_scene(image(),CameraModel.panda_agentview_default())
        self.assertGreater(m.pencil_area,100)
        self.assertGreater(m.holder_area,500)
        self.assertTrue(np.isfinite(m.pencil_xy).all())
        self.assertGreater(np.linalg.norm(m.pencil_xy-m.holder_xy),.12)

    def test_blank_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            measure_scene(np.zeros((256,256,3),np.uint8),CameraModel.panda_agentview_default())

    def test_eraser_resolves_axis_polarity(self):
        left,right=image(),image()
        cv2.rectangle(left,(46,96),(51,101),(230,90,140),-1)
        cv2.rectangle(right,(119,96),(124,101),(230,90,140),-1)
        camera=CameraModel.panda_agentview_default()
        a,b=measure_scene(left,camera),measure_scene(right,camera)
        self.assertLess(math.cos(a.pencil_yaw-b.pencil_yaw),-.95)

    def test_fsm_bounded_actions_and_timeout(self):
        now=[0.]
        t=VisionTeacher(clock=lambda:now[0])
        obs={'image':image(),'state':np.array([.4,0,.3,0,0,0,.04,-.04])}
        for _ in range(3): t.step(obs)
        self.assertEqual(t.stage,'approach')
        action=t.step(obs)
        self.assertTrue(np.isfinite(action).all())
        self.assertLessEqual(np.linalg.norm(action[:3])*.05,.015001)
        now[0]=200
        action=t.step(obs)
        self.assertIn('timeout',t.failed)
        np.testing.assert_array_equal(action[:6],0)

    def test_pencil_axis_is_gripper_y_and_becomes_vertical(self):
        t=VisionTeacher(config=TeacherConfig(verify_grasp=False))
        m=measure_scene(image(),t.camera)
        t._make_plan(m)
        grasp=t._plan[0][2]
        pencil=np.array([math.cos(m.pencil_yaw),math.sin(m.pencil_yaw),0.])
        self.assertAlmostEqual(abs(np.dot(grasp[:,1],pencil)),1.,places=7)
        self.assertAlmostEqual(np.dot(grasp[:,2],[0,0,-1]),1.,places=7)
        vertical=t._plan[4][2]
        self.assertAlmostEqual(abs(vertical[2,1]),1.,places=7)

    def test_empty_grasp_does_not_count_as_completion(self):
        now=[0.]
        t=VisionTeacher(clock=lambda:now[0])
        m=measure_scene(image(),t.camera)
        t._make_plan(m)
        t._index=2; t.stage='close'
        _,xyz,rot,_,_=t._plan[2]
        now[0]=2.
        state=np.r_[xyz,Rotation.from_matrix(rot).as_rotvec(),0.,0.]
        t.step({'state':state})
        self.assertIn('empty grasp',t.failed)
        self.assertFalse(t.done)


if __name__=='__main__':
    unittest.main()
