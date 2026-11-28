import json,torch
from pathlib import Path
from compatible import commanded_progress_error,heading_drift_cost,sample_curriculum_targets
commands=torch.tensor([[-.12,0,0],[0,.12,0],[0,-.12,0],[.25,0,0]])
zero=torch.zeros(4,3)
torch.testing.assert_close(commanded_progress_error(commands,commands),torch.zeros(4))
torch.testing.assert_close(commanded_progress_error(commands,zero),torch.ones(4))
torch.testing.assert_close(commanded_progress_error(commands,-commands),torch.full((4,),4.))
assert (commanded_progress_error(commands,commands*2)>0).all()
assert torch.isfinite(commanded_progress_error(zero,zero)).all()
assert torch.count_nonzero(commanded_progress_error(zero,zero))==0
turn=commands.clone();turn[:,2]=.2
assert torch.count_nonzero(commanded_progress_error(turn,zero))==0
angles=torch.tensor([.5,1.,1.5,2.,3.])
cost=heading_drift_cost(angles,torch.zeros(5),commands[2].repeat(5,1))
assert torch.all(cost[1:]>cost[:-1])
torch.testing.assert_close(heading_drift_cost(angles+2*torch.pi,torch.zeros(5),commands[2].repeat(5,1)),cost,atol=1e-6,rtol=1e-6)
torch.manual_seed(42);samples=sample_curriculum_targets(20000,0,"cpu")
left=int((samples[:,1]>0).sum());right=int((samples[:,1]<0).sum())
assert 1.8<right/left<2.2
result={"correct_speed_has_zero_cost":True,"stationary_wrongway_overspeed_penalized":True,
 "idle_and_turn_excluded":True,"heading_penalty_increases_beyond_57_degrees":True,
 "left_samples":left,"right_samples":right}
Path("progress_reward_test_result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
