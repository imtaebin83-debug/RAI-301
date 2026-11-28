"""Independent comparison against the unchanged official loader (CPU only)."""
import sys,json,hashlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import mujoco
from compatible import *
from mjlab.utils.lab_api.math import quat_apply_inverse
sys.path.insert(0,str(BASE/"src"))
from ros2_bridge.tools.g1_walk_policy_29dof import G1WalkPolicy29DoF
class Session:
    def run(self,names,feed):
        self.obs=next(iter(feed.values())).copy()
        return [np.full((1,29),0.123,dtype=np.float32)]

model=robot_spec().compile()
model.opt.timestep=.002
data=mujoco.MjData(model)
# Avoid ONNX dependency here; exercise actual loader's index/gain/reset/control methods.
p=G1WalkPolicy29DoF.__new__(G1WalkPolicy29DoF)
p.model=model;p.data=data;p.prefix=""
p.session=Session();p.input_name="obs";p.output_name="action"
p.stationary_idle=False;p.command_deadband=.001;p.command_slew=np.array([1.2,1.2,2.4])
p._resolve_indices();p._apply_policy_gains()
p.action=np.zeros(29);p.target_pos=DEFAULT_POS.copy();p.obs=np.zeros(98,dtype=np.float32)
p.cmd=np.zeros(3);p.target_cmd=np.zeros(3)
p._idle_pos=np.zeros(3);p._idle_quat=np.array([1.,0,0,0])
p.reset(pos=(0,0,.79))
c=CompetitionCommand.__new__(CompetitionCommand)
c._env=SimpleNamespace(common_step_counter=0,device="cpu",num_envs=1)
c.vel_command_b=torch.zeros(1,3);c.target=torch.zeros(1,3)
c.phase=torch.zeros(1,dtype=torch.float64);c.idle=torch.ones(1,dtype=torch.bool)
c.woke=torch.zeros(1,dtype=torch.bool);c.tick=torch.full((1,),-1,dtype=torch.long)
rng=np.random.default_rng(42)
previous=torch.zeros(1,29)
maximum=0;count=0
trace=[((0,0,0),5),((.6,.2,.4),45),((-.2,-.3,-.5),40),((0,0,0),50),((.3,0,0),40)]
for target,length in trace:
    p.set_command(*target);c.target[:]=torch.tensor(target)
    for step in range(length):
        c._env.common_step_counter+=1
        quat=rng.normal(size=4);quat/=np.linalg.norm(quat)
        data.qpos[3:7]=quat
        data.qpos[p.qadr]=DEFAULT_POS+rng.normal(size=29)*.1
        data.qvel[:]=rng.normal(size=model.nv)*.2
        mujoco.mj_forward(model,data)
        c.advance()
        prev=previous.clone()
        prev[c.woke|c.idle]=0
        gravity=quat_apply_inverse(torch.tensor(quat[None],dtype=torch.float32),torch.tensor([[0.,0.,-1.]]))
        obs=pack_observation(torch.tensor(data.qvel[3:6][None],dtype=torch.float32),
            gravity,c.command,c.phase,torch.tensor(data.qpos[p.qadr][None],dtype=torch.float32),
            torch.tensor(data.qvel[p.dadr][None],dtype=torch.float32),prev)
        # Advance exactly one 50 Hz controller period, with state held fixed.
        for _ in range(10): p.control(model,data)
        np.testing.assert_allclose(c.command.numpy()[0],p.cmd,atol=1e-6)
        assert bool(c.idle[0])==p._idle
        if not p._idle:
            error=float(np.max(np.abs(obs.numpy()[0]-p.obs)))
            maximum=max(maximum,error)
            np.testing.assert_allclose(obs.numpy()[0],p.obs,atol=2e-6)
            np.testing.assert_allclose(data.ctrl[p.act],DEFAULT_POS+ACTION_SCALE*.123,atol=1e-7)
            previous[:]=.123
            count+=1
        else:
            np.testing.assert_allclose(data.ctrl[p.act],DEFAULT_POS)
        phase_before=c.phase.clone()
        c.advance()
        assert torch.equal(c.phase,phase_before),"Observation read advanced phase twice"
result={"active_observations_compared":count,"max_abs_error":maximum,
    "tested":["98-input order","quaternion gravity","joint offsets","previous action",
              "command slew/reversal","stop and restart","phase wrap and suppression",
              "29-output scale and actuator mapping","idempotent phase read"]}
Path("contract_result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
