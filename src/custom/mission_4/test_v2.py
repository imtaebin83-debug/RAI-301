"""V2 CPU lifecycle and model-boundary checks; no live network or simulator."""
import io
import json
import threading
import unittest
from types import MethodType
from unittest.mock import Mock, patch
from PIL import Image

from custom.sensor_mission import Mission as SensorMission
from custom.mission_4.mission_4 import Mission4
from custom.mission_4.test_mission_4 import mission_stub, observation
from custom.mission_4 import test_mission_4 as existing_tests
from custom.vlm import _team_endpoint, _NoRedirect, classify_room, classify_views


class V2RoomTests(unittest.TestCase):
    def fixture(self):
        mission = mission_stub(begin_mission=Mock(return_value=None), stop=Mock(),
            log=Mock(), result=lambda **data:data, _travel=Mock(), _face_room=Mock(),
            _image=Mock(return_value=Image.new('RGB',(20,20))), _last_image_stamp=[2,0],
            _navigate=lambda nav,action,timeout: action(timeout))
        nav=Mock()
        nav.pose.return_value=(16.,3.59,0.)
        return mission, nav

    def test_evaluation_manifest_prepares_full_chain_without_starting_simulation(self):
        from custom.mission_4 import evaluate
        with patch.object(evaluate, 'check_gpu') as gpu, \
             patch.object(evaluate.subprocess, 'Popen') as launch:
            case = next(c for c in evaluate.cases() if c['id']=='after-m1-cone-blue')
        self.assertEqual(case['missions'], [1,2,4])
        self.assertEqual(case['m2_color'], 'blue')
        gpu.assert_not_called()
        launch.assert_not_called()

    def test_public_sensor_base_and_none_begin_contract(self):
        self.assertTrue(issubclass(Mission4, SensorMission))
        mission, nav=self.fixture()
        with patch('custom.mission_4.mission_4.Navigator',return_value=nav) as factory, \
             patch('custom.mission_4.mission_4.classify_views',return_value=[
                 observation(label) for label in ('medical','cafeteria','classroom')]), \
             patch('custom.mission_4.mission_4.time.sleep'), \
             patch.dict('os.environ',{},clear=True):
            result=Mission4.run(mission)
        factory.assert_called_once_with(mission,None)
        self.assertNotIn('error',result)
        self.assertEqual(result['rooms'],['medical','cafeteria','classroom'])

    def test_respawn_invalidates_images_labels_and_delivery_at_each_phase(self):
        for phase in ('begin','image','inference','crop','return','entry'):
            with self.subTest(phase=phase):
                mission,nav=self.fixture()
                labels=[observation(label) for label in ('medical','cafeteria','classroom')]
                def respawn(value):
                    mission.respawn_count+=1
                    return value
                classify=Mock(return_value=labels)
                if phase=='begin':
                    mission.begin_mission.side_effect=lambda *a,**k:respawn(None)
                if phase=='image':
                    mission._image.side_effect=lambda *a,**k:respawn(Image.new('RGB',(20,20)))
                if phase=='inference':
                    classify.side_effect=lambda *a,**k:respawn(labels)
                if phase=='crop':
                    calls=[0]
                    def infer(*a,**k):
                        calls[0]+=1
                        if calls[0]==1:
                            return [{'label':'medical','scores':dict.fromkeys(
                                ('medical','cafeteria','classroom'),-5.)}]*3
                        return respawn(labels)
                    classify.side_effect=infer
                if phase=='return':
                    calls=[0]
                    def travel(*a):
                        calls[0]+=1
                        if calls[0]==4:
                            respawn(None)
                    mission._travel.side_effect=travel
                if phase=='entry':
                    nav.goto.side_effect=lambda *a,**k:respawn(True)
                with patch('custom.mission_4.mission_4.Navigator',return_value=nav), \
                     patch('custom.mission_4.mission_4.classify_views',classify), \
                     patch('custom.mission_4.mission_4.time.sleep'), \
                     patch.dict('os.environ',{},clear=True):
                    result=Mission4.run(mission)
                self.assertTrue(result['restart_required'])
                self.assertTrue(result['observations_invalidated'])
                for field in ('rooms','views','vlm_observations','vlm_retries'):
                    self.assertEqual(result[field],[])
                self.assertIsNone(result['room_assignment'])
                if phase!='entry':
                    nav.goto.assert_not_called()
                if phase in ('begin','image'):
                    classify.assert_not_called()
                mission.stop.assert_called()

    def test_respawn_during_pose_recovery_is_not_treated_as_fresh_localization(self):
        mission,nav=self.fixture()
        mission._run_respawn_count=0
        mission._recovery_attempts=0
        mission._pose_recoveries=[]
        mission.latest=Mock(return_value=None)
        action=Mock(side_effect=RuntimeError('Stale localization: injected'))
        def sleep(_):
            mission.respawn_count+=1
        with patch('custom.mission_4.mission_4.time.sleep',side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError,'respawn'):
                Mission4._navigate(mission,nav,action,20.)
        action.assert_called_once()
        self.assertFalse(mission._pose_recoveries[0]['recovered'])

    def test_camera_discards_fresh_frames_when_respawn_occurs(self):
        mission=mission_stub(stop=Mock())
        mission._run_respawn_count=0
        camera=existing_tests.CameraTests()
        calls=[0]
        def latest(*a,**k):
            calls[0]+=1
            if calls[0]==3:
                mission.respawn_count+=1
            return camera.image(calls[0])
        mission.latest=latest
        with patch('custom.mission_4.mission_4.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'respawn'):
                Mission4._image(mission,timeout=1.)

    def test_public_notification_stops_and_blocks_subsequent_motion(self):
        node=Mission4.__new__(Mission4)
        node._sensor_cv=threading.Condition()
        node._sensor_messages={}
        node._respawn_count=node._run_respawn_count=0
        with patch.object(SensorMission,'cmd_vel',autospec=True) as publish:
            node.cmd_vel(.1,0,0)
            node._on_public_respawn(object())
            with self.assertRaisesRegex(RuntimeError,'respawn'):
                node.cmd_vel(.1,0,0)
            node.stop()
        self.assertTrue(all(call.args[1:]==(0.,0.,0.) for call in publish.call_args_list[1:]))


class TeamModelTests(unittest.TestCase):
    def test_local_endpoint_shapes(self):
        for url in ('http://127.0.0.1:8000/v1','http://100.77.17.47:8000/v1/',
                    'https://192.168.1.5/v1','http://[::1]:8000/v1','http://localhost:8000/v1'):
            self.assertTrue(_team_endpoint(url).endswith('/v1'))

    def test_external_or_ambiguous_endpoints_rejected_before_any_io(self):
        for url in ('https://api.openai.com/v1','https://8.8.8.8/v1',
                    'http://169.254.169.254/v1','http://team.example/v1',
                    'file:///tmp/v1','http://user:pass@127.0.0.1/v1',
                    'http://127.0.0.1/v1?forward=external','http://127.0.0.1/v1#x',
                    'http://127.0.0.1:0/v1'):
            with self.subTest(url=url), \
                 patch.dict('os.environ',{'HAIC_VLM_BASE_URL':url},clear=True), \
                 patch('custom.vlm.urllib.request.build_opener') as opener:
                image=Mock()
                with self.assertRaises(ValueError):
                    classify_room(image)
                image.save.assert_not_called()
                opener.assert_not_called()

    def test_team_server_does_not_inherit_commercial_key_or_proxies(self):
        response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'medical'}}]}).encode())
        opener=Mock()
        opener.open.return_value=response
        with patch.dict('os.environ',{'HAIC_VLM_BASE_URL':'http://100.77.17.47:8000/v1',
                    'HAIC_VLM_MODEL':'team-vlm','OPENAI_API_KEY':'unused-test-value',
                    'HTTPS_PROXY':'http://proxy.invalid'},clear=True), \
             patch('custom.vlm.urllib.request.build_opener',return_value=opener) as build:
            result=classify_room(Image.new('RGB',(2,2)),timeout=2.)
        request=opener.open.call_args.args[0]
        self.assertIsNone(request.get_header('Authorization'))
        self.assertEqual(build.call_args.args[0].proxies,{})
        self.assertIsInstance(build.call_args.args[1],_NoRedirect)
        self.assertEqual(result['backend'],'team_local_api')
        self.assertEqual(result['label'],'medical')

    def test_redirects_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'redirect'):
            _NoRedirect().redirect_request(None,None,302,'',{},'https://public.example')

    def test_single_image_local_worker_is_cpu_only(self):
        with patch.dict('os.environ',{},clear=True), \
             patch('custom.vlm.subprocess.run',return_value=Mock(
                 returncode=0,stdout='{"label":"medical"}')) as run:
            self.assertEqual(classify_room(Image.new('RGB',(2,2)))['label'],'medical')
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')


if __name__=='__main__':
    unittest.main()
