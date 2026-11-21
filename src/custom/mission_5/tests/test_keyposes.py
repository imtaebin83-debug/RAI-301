"""Synthetic numerical tests only; these fixtures are not robot demonstrations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from custom.mission_5.keypose_policy import Policy, materialize, validate_model, yaw_rotation
from custom.mission_5.train_keyposes import canonical_pose, fit_model


def synthetic_episodes():
    rows = []
    for i, yaw in enumerate((-.5, .2, .8, -.2)):
        scene = {'pencil_xy': [.3 + .03*i, -.15 + .02*i],
                 'holder_xy': [.4 - .015*i, .2 + .03*i], 'pencil_yaw': yaw}
        phases = []
        for name, reference, xyz, grip in [('sample-a', 'pencil', [.004, -.003, .31], -1.),
                                           ('sample-b', 'holder', [-.002, .005, .28], 1.)]:
            R = yaw_rotation(yaw)
            pos = np.r_[scene[reference + '_xy'], 0.] + R @ xyz
            orientation = R @ Rotation.from_euler('x', 2.).as_matrix()
            state = np.r_[pos, Rotation.from_matrix(orientation).as_rotvec(), .02, -.02]
            phases.append({'name': name, 'state': state, 'grip': grip, 'duration': 1. + .1*i})
        rows.append({'scene': scene, 'phases': phases, 'file': f'synthetic{i}.h5',
                     'sha256': str(i)*64, 'fps': 10.,
                     'camera': {'fovy':46.,'pos':[1.28,0,.92],
                                'quat':[.6633178,.24496836,.24496836,.6633178]}})
    return rows


class KeyposeLearningTests(unittest.TestCase):
    def test_reference_frames_and_arbitrary_phase_names_are_learned(self):
        model = fit_model(synthetic_episodes(), manifest_sha256='synthetic-test')
        self.assertEqual([p['name'] for p in model['phases']], ['sample-a','sample-b'])
        self.assertEqual([p['reference'] for p in model['phases']], ['pencil','holder'])
        self.assertEqual([p['grip'] for p in model['phases']], [-1.,1.])
        self.assertAlmostEqual(model['phases'][0]['minimum_seconds'],1.15)

    def test_object_relative_prediction_generalizes_rigid_scene_transform(self):
        rows = synthetic_episodes()
        model = fit_model(rows,manifest_sha256='synthetic-test')
        scene={'pencil_xy':[-.1,.7],'holder_xy':[.6,-.3],'pencil_yaw':1.2}
        predicted=materialize(model['phases'][0],scene)
        expected=np.r_[scene['pencil_xy'],0.]+yaw_rotation(1.2)@np.array([.004,-.003,.31])
        np.testing.assert_allclose(predicted[:3,3],expected,atol=1e-8)

    def test_changed_training_endpoints_change_executed_action(self):
        rows=synthetic_episodes()
        first=fit_model(rows,manifest_sha256='synthetic-test')
        for row in rows:
            row['phases'][0]['state'][2]+=.05
        second=fit_model(rows,manifest_sha256='synthetic-test')
        scene={'pencil_xy':[.3,-.15],'holder_xy':[.4,.2],'pencil_yaw':0.}
        target=materialize(first['phases'][0],scene)
        state=np.r_[target[:3,3]+[0,0,.025],Rotation.from_matrix(target[:3,:3]).as_rotvec(),.02,-.02]
        with tempfile.TemporaryDirectory() as directory:
            actions=[]
            for n,model in enumerate((first,second)):
                path=Path(directory)/f'm{n}.json';path.write_text(json.dumps(model))
                p=Policy(path);p._scene=scene;p._start=p._stage_started=0.;p.stage='sample-a'
                actions.append(p.act({'state':state,'time':.1})['actions'][0])
        self.assertLess(actions[0][2],0)
        self.assertGreater(actions[1][2],0)

    def test_no_checkpoint_or_missing_provenance_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            Policy('/this/checkpoint/does/not/exist.json')
        model=fit_model(synthetic_episodes(),manifest_sha256='synthetic-test')
        model['training']['episode_sha256']={}
        with self.assertRaises(ValueError):validate_model(model)

    def test_nonfinite_learned_pose_is_rejected(self):
        model=fit_model(synthetic_episodes(),manifest_sha256='synthetic-test')
        model['phases'][0]['offset_xyz'][2]=float('nan')
        with self.assertRaises(ValueError):validate_model(model)

    def test_incompatible_phase_sequences_cannot_be_averaged(self):
        rows=synthetic_episodes();rows[-1]['phases'].reverse()
        with self.assertRaises(ValueError):fit_model(rows,manifest_sha256='synthetic-test')


if __name__=='__main__':unittest.main()
