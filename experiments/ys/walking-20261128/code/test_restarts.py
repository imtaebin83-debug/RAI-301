from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,torch,mujoco
from compatible import *
rng=np.random.default_rng(2026)
errors=[]
for _ in range(200):
 q=rng.normal(size=(2,4));q/=np.linalg.norm(q,axis=1)[:,None]
 pos=rng.normal(size=3)*.01;held=rng.normal(size=3)*.01;vel=rng.normal(size=6)*.2
 rotation=np.zeros(3);mujoco.mju_subQuat(rotation,q[0],q[1])
 expected=np.r_[np.clip(np.array([6000,6000,2500])*(held-pos)-np.array([300,300,220])*vel[:3],-5000,5000),
     np.clip(1000*rotation-80*vel[3:],-1000,1000)]
 actual=official_idle_wrench(*[torch.tensor(x[None],dtype=torch.float64) for x in [pos,q[1],vel,held,q[0]]],torch.tensor([True])).numpy()[0]
 np.testing.assert_allclose(actual,expected,atol=1e-9)
 errors.append(float(np.max(np.abs(actual-expected))))
 zero=official_idle_wrench(*[torch.tensor(x[None],dtype=torch.float64) for x in [pos,q[1],vel,held,q[0]]],torch.tensor([False]))
 assert torch.count_nonzero(zero)==0
command=torch.tensor([[0.,.12,0],[0.,.12,.2],[.2,0,0]])
cost=heading_drift_cost(torch.tensor([.5,.5,.5]),torch.zeros(3),command)
torch.testing.assert_close(cost,torch.tensor([2*(1-np.cos(.5)),0.,2*(1-np.cos(.5))],dtype=cost.dtype))
# Verify the actual command scheduling alternates move/idle and reuses direction.
original=CompetitionCommand._resample_command
CompetitionCommand._resample_command=lambda self,ids:None
try:
 c=WalkingCurriculumCommand.__new__(WalkingCurriculumCommand)
 c._env=SimpleNamespace(num_envs=200,device="cpu")
 c.target=torch.zeros(200,3);c.time_left=torch.zeros(200)
 c.is_standing_env=torch.zeros(200,dtype=torch.bool)
 ids=torch.arange(200)
 c._resample_command(ids);assert (c.target==0).all() and ((c.time_left>=1)&(c.time_left<=2)).all()
 c._resample_command(ids);first=c.target.clone();assert ((c.time_left>=3.5)&(c.time_left<=4.5)).all()
 c.heading_pending[:]=False
 c._resample_command(ids);assert (c.target==0).all() and ((c.time_left>=1)&(c.time_left<=2)).all()
 assert not c.heading_pending.any()
 c._resample_command(ids);torch.testing.assert_close(c.target,first)
 assert not c.heading_pending.any()
 c._resample_command(ids);assert not c.heading_pending.any()
 c._resample_command(ids);assert c.heading_pending.all()
finally:CompetitionCommand._resample_command=original
cfg=make_cfg(2)
result={"official_idle_wrench_parity_samples":200,"maximum_abs_error":max(errors),
    "active_force_clear_pass":True,"heading_reward_pass":True,"repeated_restart_schedule_pass":True,"heading_reference_survives_pause_and_restart":True}
Path("restart_test_result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
