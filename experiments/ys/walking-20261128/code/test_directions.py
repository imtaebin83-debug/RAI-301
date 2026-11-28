import json
from pathlib import Path
import torch
from compatible import sample_curriculum_targets,make_cfg
from curriculum_gate import next_stage,skill_gate
torch.manual_seed(42)
stats={}
for stage,limit in [(0,.12),(1,.18),(2,.25)]:
 x=sample_curriculum_targets(20000,stage,"cpu")
 assert torch.isfinite(x).all()
 assert (x[:,0]<0).any() and (x[:,1]>0).any() and (x[:,1]<0).any()
 assert x[x[:,0]<0,0].abs().max()<=limit
 assert x[:,1].abs().max()<=limit
 forward=float((x[:,0]>0).float().mean())
 assert .38<forward<.42
 stats[stage]={"forward":forward,"backward":float((x[:,0]<0).float().mean()),
     "lateral":float((x[:,1]!=0).float().mean())}
assert next_stage(0,True,True,False)==0
assert next_stage(0,True,False,True)==1
assert next_stage(1,False,True,True)==0
assert next_stage(1,True,True,True)==2
# A stationary or wrong-way policy cannot pass the direction gate.
for direction in [(-.12,0,0),(0,.12,0),(0,-.12,0)]:
 for sign in (0,-1):
  row={"policy":"candidate","failure":None,"linear_velocity_rmse":.01,"yaw_velocity_rmse":.01,
      "max_tilt_degrees":1,"samples":[{"command":list(direction),"position":[0,0,.78]},
      {"command":list(direction),"position":[direction[0]*3*sign,direction[1]*3*sign,.78]}]}
  assert not skill_gate([row],direction)[0]
cfg=make_cfg(2)
Path("direction_test_result.json").write_text(json.dumps({"sampling":stats,"stage_gates":True,"stationary_wrongway_rejected":True},indent=2))
print("Direction curriculum and gate tests passed",stats)
