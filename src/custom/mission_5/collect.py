"""Development-only, sensor-only demonstration collection with official labels.

Run beside your own reserved main.py --sim-only. The teacher receives only the
two RGB images and robot state. Referee results are read only AFTER rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import h5py
import numpy as np
import yaml
from PIL import Image as PILImage

def panda_calibration():
    """Public camera metadata moved into custom/robot.yaml in starter v2."""
    path=Path(__file__).resolve().parents[1]/'robot.yaml'
    return path,yaml.safe_load(path.read_text())['sensors']['panda']


def panda_camera(config):
    """Build the raw256 camera model from public configuration, not CameraInfo."""
    fovy=float(config['fovy'])
    if not math.isfinite(fovy) or not 0<fovy<180:
        raise ValueError('Panda camera fovy must be finite and between 0 and 180 degrees')
    from custom.mission_5.perception import CameraModel
    focal=128./math.tan(math.radians(fovy)/2.)
    intrinsics=np.array([[focal,0.,127.5],[0.,focal,127.5],[0.,0.,1.]])
    return CameraModel.from_pose(intrinsics,config['pos'],config['quat'])


class Writer:
    def __init__(self,path,fps,placement):
        calibration,panda=panda_calibration()
        self.path=Path(path)
        self.temporary=self.path.with_name('.'+self.path.stem+'.recording.h5')
        self.file=h5py.File(self.temporary,'w')
        self.file.attrs.update(schema='m5-two-camera-episodes-v1',fps=fps,
            placement_id=placement,success=-1,state_frame='panda_base',action_frame='panda_base',
            action_position_scale_m=.05,action_rotation_scale_rad=.5,
            policy_inputs='two RGB cameras and robot proprioception only')
        self.file.attrs['camera_calibration_sha256']=hashlib.sha256(calibration.read_bytes()).hexdigest()
        self.file.attrs['camera_metadata_json']=json.dumps(panda)
        self.count=0
    def append(self,obs,action):
        row={'time':np.asarray(obs['time'],np.float64),
             'observation/image':obs['image'],'observation/wrist_image':obs['wrist_image'],
             'observation/state':np.asarray(obs['state'],np.float32),'actions':np.asarray(action,np.float32)}
        for name,value in row.items():
            value=np.asarray(value)
            if name not in self.file:
                self.file.create_dataset(name,shape=(0,*value.shape),maxshape=(None,*value.shape),
                    dtype=value.dtype,chunks=(1,*value.shape),compression='lzf' if value.size>32 else None)
            dataset=self.file[name]
            dataset.resize(self.count+1,axis=0)
            dataset[self.count]=value
        self.count+=1
    def finish(self,verdict,reason):
        success=bool(verdict.get('success',False))
        self.file.attrs.update(success=int(success),official_success=success,finish_reason=reason)
        self.file.close()
        self.temporary.replace(self.path)


def manual_action(key,grip):
    action=np.zeros(7,np.float32)
    moves={'w':(0,1),'s':(0,-1),'a':(1,1),'d':(1,-1),'t':(2,1),'b':(2,-1),
           '1':(3,1),'2':(3,-1),'3':(4,1),'4':(4,-1),'5':(5,1),'6':(5,-1)}
    if key in moves:
        axis,sign=moves[key];action[axis]=sign*.08
    if key=='g':grip=-grip
    action[6]=grip
    return action,grip


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['teacher','manual','observe'],default='teacher')
    parser.add_argument('--seconds',type=float,default=100)
    parser.add_argument('--fps',type=float,default=10)
    parser.add_argument('--retry',action='store_true',
        help='Unsupported in starter v2: restart your simulator for a fresh placement')
    args=parser.parse_args(argv)
    if args.retry:
        parser.error('--retry is unavailable: starter v2 removed /panda/retry. '
                     'Start a fresh simulator for each new collection placement.')
    if not 1<=args.fps<=20 or not 0<args.seconds<=300:parser.error('invalid seconds/fps')
    if args.mode=='manual' and not (os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')):
        parser.error('Manual collection needs an Ubuntu desktop terminal/display')
    return args


def main(argv=None):
    args=parse_args(argv)
    # Parse unsupported reset requests before importing/initializing ROS.
    from mission_api import MissionRunner
    from custom.sensor_mission import Mission
    from custom.mission_5.ros_io import PandaROSIO

    class CollectorMission(Mission):
        MISSION=5

    args.out.mkdir(parents=True,exist_ok=True)
    episode_id=f'{time.time_ns()}'
    file=args.out/f'episode_{episode_id}.h5'
    events=[]; writer=None; interface=None
    with MissionRunner('m5_collection_evaluator') as runner:
        runner.api.wait_for_sim(timeout=40)
        node=runner.start(CollectorMission)
        node.begin_mission(5,timeout=15)
        interface=PandaROSIO(node)
        from referee import Referee
        referee=Referee(None)
        referee.begin(5)
        reason='time_limit'; observations=0
        try:
            initial=interface.observation(timeout=10)
            if args.mode=='observe':
                print(json.dumps({'state':initial['state'].tolist(),'time':initial['time'],
                                  'base_parent':interface.base_parent},indent=2))
                for key in ('image','wrist_image'):PILImage.fromarray(initial[key]).save(args.out/(key+'.png'))
                return
            teacher=None; sim_clock=[float(initial['time'])]
            if args.mode=='teacher':
                from custom.mission_5.vision_teacher import VisionTeacher
                _,panda=panda_calibration()
                camera=panda_camera(panda['agentview'])
                teacher=VisionTeacher(camera=camera,clock=lambda:sim_clock[0])
            # Group repeated identical sensor-observed placements together.
            fingerprint=hashlib.sha256(initial['image'].tobytes()).hexdigest()
            writer=Writer(file,args.fps,'sensor-session:'+fingerprint)
            start=float(initial['time']); previous=None; wall_end=time.monotonic()+3*args.seconds+30
            stage=None; grip=-1.; done_at=None
            while time.monotonic()<wall_end:
                obs=interface.observation(timeout=3.)
                now=float(obs['time'])
                if now-start>args.seconds:break
                if previous is not None and now-previous<.8/args.fps:
                    time.sleep(.005);continue
                sim_clock[0]=now
                if args.mode=='teacher':
                    action=teacher.step(obs)
                    if teacher.stage!=stage:
                        stage=teacher.stage
                        debug={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in teacher.debug.items()}
                        row={'seconds':now-start,'stage':stage,'state':obs['state'].tolist(),'debug':debug}
                        events.append(row); print(json.dumps(row),flush=True)
                        PILImage.fromarray(obs['image']).save(args.out/f'{episode_id}_{stage}.png')
                        if stage=='approach':
                            measure=teacher.initial
                            values=np.r_[measure['pencil_xy'],measure['holder_xy']]
                            writer.file.attrs['placement_id']='vision:'+hashlib.sha256(np.round(values,2).tobytes()).hexdigest()
                    if teacher.done:
                        reason='teacher_complete'
                        if done_at is None:done_at=now
                        if now-done_at>=2.:break
                    if teacher.failed:reason=str(teacher.failed);break
                else:
                    import cv2
                    canvas=np.concatenate([obs['image'],obs['wrist_image']],axis=1)
                    cv2.imshow('M5 W/S X A/D Y T/B Z 1-6 rotate G grip P finish Q abort',canvas[:,:,::-1])
                    raw=cv2.waitKey(1)&255;key=chr(raw) if raw!=255 else ''
                    if key in ('p','q'):reason='operator_finished' if key=='p' else 'operator_aborted';break
                    action,grip=manual_action(key,grip)
                action=np.asarray(action,np.float32)
                if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Invalid teacher action')
                writer.append(obs,action)
                interface.command(action)
                observations+=1;previous=now
            interface.hold()
        except Exception as error:
            reason=f'{type(error).__name__}: {error}'
            print(reason,flush=True)
        finally:
            if interface is not None:interface.hold()
        result=node.result(collection=True,mode=args.mode,reason=reason,frames=observations)
        record=referee.finish(runner.api,result)
        if writer is not None:writer.finish(result.judge or {},reason)
        report={'hdf5':file.name,'reason':reason,'frames':observations,'events':events,
                'official_record':record,'policy_inputs':['image','wrist_image','state']}
        (args.out/f'{file.stem}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'hdf5':str(file),'reason':reason,'frames':observations,'success':record.get('success')}),flush=True)
        runner.finish(node)


if __name__=='__main__':main()
