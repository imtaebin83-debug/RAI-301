import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import mujoco
import numpy as np
import torch
from compatible import make_cfg,JOINTS,DEFAULT_POS,ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
parser=argparse.ArgumentParser()
parser.add_argument("--iterations",type=int,default=1000)
parser.add_argument("--envs",type=int,default=512)
parser.add_argument("--full-training",action="store_true",help="Explicitly allow more than 500 iterations after reviewing preflight evidence")
parser.add_argument("--initialize-actor",type=Path,required=True)
args=parser.parse_args()
if args.iterations>500 and not args.full_training:
    parser.error("Runs above 500 iterations require --full-training after reviewing preflight evidence.")
if args.iterations<1 or args.envs<1:
    parser.error("iterations and envs must be positive")
from preflight import verify_sources
verify_sources()
out=Path.cwd()
if list(out.glob("iteration_*")) or (out/"small_checkpoint.pt").exists() or (out/"small_policy.onnx").exists():
    parser.error("Use a fresh output directory to preserve existing checkpoints.")
torch.manual_seed(42)
start=time.monotonic()
env=ManagerBasedRlEnv(make_cfg(args.envs),device="cuda:0")
assert tuple(env.scene["robot"].joint_names)==tuple(JOINTS)
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
obs=wrapped.get_observations()
assert obs["actor"].shape==(args.envs,98) and torch.isfinite(obs["actor"]).all()
# Independently compare the attached training robot with the original robot.
from compatible import robot_spec
reference=robot_spec().compile()
model=env.sim.mj_model
checks={}
for kind,fields in [(mujoco.mjtObj.mjOBJ_BODY,("body_mass","body_inertia","body_ipos","body_iquat")),
                    (mujoco.mjtObj.mjOBJ_JOINT,("jnt_range","jnt_axis","jnt_pos")),
                    (mujoco.mjtObj.mjOBJ_ACTUATOR,("actuator_gainprm","actuator_biasprm","actuator_forcerange"))]:
    for i in range({mujoco.mjtObj.mjOBJ_BODY:reference.nbody,mujoco.mjtObj.mjOBJ_JOINT:reference.njnt,mujoco.mjtObj.mjOBJ_ACTUATOR:reference.nu}[kind]):
        name=mujoco.mj_id2name(reference,kind,i)
        if not name or name=="world": continue
        j=mujoco.mj_name2id(model,kind,"robot/"+name)
        assert j>=0,(name,"missing in training")
        for field in fields:
            np.testing.assert_allclose(getattr(model,field)[j],getattr(reference,field)[i],atol=1e-9)
            checks[field]=checks.get(field,0)+1
# Test frame conversion against raw free-joint angular velocity at randomized yaw.
robot=env.scene["robot"]
angular_ids=robot.indexing.free_joint_v_adr[3:6]
env.sim.data.qvel[:,angular_ids]=torch.tensor((0.2,-0.3,0.4),device=env.device)
env.sim.forward()
raw=env.sim.data.qvel[:,angular_ids]
np.testing.assert_allclose(robot.data.root_link_ang_vel_b.cpu(),raw.cpu(),atol=1e-6)
wrapped.reset()
(out/"physical_parity.json").write_text(json.dumps(checks,indent=2))
(out/"source_config.py.txt").write_text(Path(__file__).with_name("compatible.py").read_text())
physics={key:getattr(model.opt,key).tolist() if isinstance(getattr(model.opt,key),np.ndarray) else float(getattr(model.opt,key)) for key in ("timestep","gravity","impratio","iterations","ls_iterations","noslip_iterations","integrator","cone","jacobian")}
(out/"training_physics.json").write_text(json.dumps(physics,indent=2))
action=env.action_manager.get_term("joint_pos")
np.testing.assert_allclose(action.offset[0].cpu(),DEFAULT_POS,atol=1e-7)
np.testing.assert_allclose(action.scale[0].cpu(),ACTION_SCALE,atol=1e-7)
# Exercise idle assistance and active-force clearing with the actual GPU arrays.
command=env.command_manager.get_term("twist")
command.idle[:]=True
act=env.action_manager.get_term("joint_pos")
act.apply_actions()
root_ids=env.scene["robot"].indexing.free_joint_v_adr
env.sim.data.qvel[:,root_ids[0]]=0.01
act.apply_actions()
assert torch.all(env.sim.data.qfrc_applied[:,root_ids[0]]<0)
command.idle[:]=False
act.apply_actions()
assert torch.count_nonzero(env.sim.data.qfrc_applied[:,root_ids])==0
wrapped.reset()
(out/"idle_gpu_bridge.json").write_text(json.dumps({"idle_force_applied":True,"active_force_cleared":True},indent=2))
agent=unitree_g1_ppo_runner_cfg()
agent.max_iterations=args.iterations
agent.algorithm.learning_rate=5e-6
agent.algorithm.schedule="fixed"
agent.algorithm.desired_kl=.005
agent.algorithm.clip_param=.1
agent.algorithm.entropy_coef=.005
agent.save_interval=1
agent.logger="tensorboard"
agent.upload_model=False
(out/"agent.json").write_text(json.dumps(asdict(agent),indent=2))
import subprocess,hashlib,os
milestones={100,250,500,args.iterations}
class PilotRunner(MjlabOnPolicyRunner):
    def save(self,path,infos=None):
        count=self.current_learning_iteration+1
        if count not in milestones or count in self.completed:
            return
        checkpoint=out/("iteration_%04d"%count)
        checkpoint.mkdir(exist_ok=False)
        super().save(str(checkpoint/"checkpoint.pt"),infos)
        policy=self.alg.get_policy()
        assert all(torch.isfinite(v).all() for v in policy.state_dict().values())
        self.export_policy_to_onnx(str(checkpoint),"small_policy.onnx")
        module=policy.as_onnx(verbose=False).cpu().eval()
        generator=torch.Generator().manual_seed(123)
        x=torch.randn((100,98),generator=generator)*.25
        x[:,5]=-1
        with torch.no_grad():
            expected=torch.cat([module(row[None]) for row in x]).numpy()
        np.savez(checkpoint/"export_parity.npz",inputs=x.numpy(),expected=expected)
        subprocess.run(["/home/aril/miniconda3/envs/haic_rai301_v2/bin/python","-u",str(Path(__file__).with_name("evaluate_pilot.py"))],
            cwd=checkpoint,check=True,timeout=600,env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"})
        subprocess.run(["/home/aril/miniconda3/envs/haic_rai301_v2/bin/python","-u",
            str(Path(__file__).with_name("evaluate_forward.py"))],cwd=checkpoint,check=True,
            timeout=600,env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"})
        from curriculum_gate import forward_gate,next_stage,skill_gate
        passed,details=forward_gate(json.loads((checkpoint/"forward/rollout_result.json").read_text()))
        command=self.env.unwrapped.command_manager.get_term("twist")
        before=getattr(command,"stage",0)
        broad=json.loads((checkpoint/"summary.json").read_text())["candidate"]
        broad_ok=(broad["passed"]==3 and broad["mean_linear_velocity_rmse"]<=.18 and broad["max_tilt_degrees"]<25)
        skill_results={}
        for skill,direction in {"backward":(-.12,0,0),"left":(0,.12,0),"right":(0,-.12,0)}.items():
            subprocess.run(["/home/aril/miniconda3/envs/haic_rai301_v2/bin/python","-u",
                str(Path(__file__).with_name("evaluate_skill.py")),skill],cwd=checkpoint,
                check=True,timeout=600,env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"})
            ok,metrics=skill_gate(json.loads((checkpoint/"skills"/skill/"rollout_result.json").read_text()),direction)
            skill_results[skill]={"pass":ok,"metrics":metrics}
        skills_ok=all(x["pass"] for x in skill_results.values())
        command.stage=next_stage(before,passed,broad_ok,skills_ok)
        (checkpoint/"curriculum_gate.json").write_text(json.dumps({"before":before,"after":command.stage,
            "forward_passed":passed,"broad_passed":broad_ok,"details":details,"skills":skill_results},indent=2))
        if count in (250,500) or args.iterations<=10:
            subprocess.run(["/home/aril/miniconda3/envs/haic_rai301_v2/bin/python","-u",str(Path(__file__).with_name("evaluate_extended.py"))],
                cwd=checkpoint,check=True,timeout=600,
                env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
                     "YS_EVAL_SUBDIR":"extended"})
        verify_sources()
        elapsed=time.monotonic()-self.started
        result={"completed_iterations":count,"seconds_including_evaluations":elapsed,
            "checkpoint":str(checkpoint),"export":json.loads((checkpoint/"export_result.json").read_text()),
            "evaluations":json.loads((checkpoint/"summary.json").read_text())}
        (checkpoint/"milestone.json").write_text(json.dumps(result,indent=2))
        if out==Path(__file__).resolve().parent:
            from assess import assess
            assess()
        self.completed.append(count)
        (out/"progress.json").write_text(json.dumps(result,indent=2))
        print("PILOT_MILESTONE "+json.dumps(result),flush=True)
        self.broad_failures=0 if broad_ok else self.broad_failures+1
        if self.broad_failures>=2:
            (out/"early_stop.json").write_text(json.dumps({"reason":"Two consecutive mixed-walking regressions","iteration":count},indent=2))
            raise RuntimeError("Safety stop: two consecutive mixed-walking regressions")
runner=PilotRunner(wrapped,asdict(agent),log_dir=str(out/"logs"),device="cuda:0")
# Continue our own checkpoint, including critic and optimizer state.
seed_checkpoint=torch.load(args.initialize_actor,map_location="cpu",weights_only=False)
runner.load(str(args.initialize_actor),map_location="cuda:0")
for key,value in seed_checkpoint["actor_state_dict"].items():
    assert torch.equal(value,runner.alg.get_policy().state_dict()[key].detach().cpu())
for key,value in seed_checkpoint["critic_state_dict"].items():
    assert torch.equal(value,runner.alg.critic.state_dict()[key].detach().cpu())
runner.current_learning_iteration=0
env.common_step_counter=0
runner.alg.learning_rate=agent.algorithm.learning_rate
for group in runner.alg.optimizer.param_groups:
    group["lr"]=agent.algorithm.learning_rate
wrapped.reset()
(out/"initialization.json").write_text(json.dumps({
    "checkpoint":str(args.initialize_actor),
    "sha256":hashlib.sha256(args.initialize_actor.read_bytes()).hexdigest(),
    "restored":"actor, critic, normalizers, optimizer",
    "additional_iterations":args.iterations,"learning_rate":runner.alg.learning_rate,
    "environment":"fresh reset; curriculum starts with gentle direction practice"},indent=2))
runner.completed=[]
runner.broad_failures=0
runner.started=time.monotonic()
runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
verify_sources()
(out/"done.json").write_text(json.dumps({"iterations":args.iterations,"milestones":runner.completed,
    "wall_seconds":time.monotonic()-start,"working_policy_unchanged":True},indent=2))
env.close()
