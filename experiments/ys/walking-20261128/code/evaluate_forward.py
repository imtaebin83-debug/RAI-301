"""Use the official MuJoCo environment and unmodified loader, no policy replacement."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
import mujoco
BASE=Path("/home/aril/rai-worktrees/ys-all-v2-20261127")
sys.path.insert(0,str(BASE/"src"))
from ros2_bridge.tools.interactive_sim import InteractiveSim
from ros2_bridge.tools.g1_walk_policy_29dof import G1WalkPolicy29DoF
checkpoint=Path.cwd()
root=checkpoint/"forward"
root.mkdir(exist_ok=False)
for name in ("small_policy.onnx","export_parity.npz"):
    (root/name).symlink_to(checkpoint/name)
data=np.load(root/"export_parity.npz")
session=ort.InferenceSession(str(root/"small_policy.onnx"),providers=["CPUExecutionProvider"])
assert session.get_inputs()[0].shape==[1,98]
assert session.get_outputs()[0].shape==[1,29]
actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in data["inputs"]])
np.testing.assert_allclose(actual,data["expected"],atol=2e-5,rtol=2e-5)
parity={"samples":len(actual),"max_abs_error":float(np.max(np.abs(actual-data["expected"]))),
        "input":[1,98],"output":[1,29],"provider":session.get_providers()}
(root/"export_result.json").write_text(json.dumps(parity,indent=2))
print("EXPORT",parity,flush=True)
sim=InteractiveSim(BASE/"src/environment/test_zone/world/layout.json",static_scene=True)
spawn=sim.spawn_robot("g1_sensors",pos=(16,11),yaw=0,keyframe=None,hold_keyframe=False,prefix="ys_training_test")
model=sim.model
initial_qpos=sim.data.qpos.copy()
physics={key:getattr(model.opt,key).tolist() if isinstance(getattr(model.opt,key),np.ndarray) else int(getattr(model.opt,key)) if key in ("iterations","ls_iterations","noslip_iterations","integrator","cone","jacobian","enableflags") else float(getattr(model.opt,key)) for key in ("timestep","gravity","impratio","iterations","ls_iterations","noslip_iterations","integrator","cone","jacobian","enableflags")}
(root/"competition_physics.json").write_text(json.dumps(physics,indent=2))
supplied=BASE/"src/environment/robots/control/g1/walking/g1_29dof_velocity.onnx"
cases=[(label,path,4,yaw) for yaw in (-.2,0.,.2)
       for label,path in (("supplied",supplied),("candidate",root/"small_policy.onnx"))]
results=[]
for label,path,noslip,yaw in cases:
    model.opt.noslip_iterations=noslip
    d=mujoco.MjData(model)
    d.qpos[:]=initial_qpos
    p=G1WalkPolicy29DoF(model,d,path,prefix=spawn.prefix)
    p.reset(pos=spawn.pos,yaw=yaw)
    mujoco.mj_forward(model,d)
    trace=[((0,0,0),1),((.25,0,0),5),((0,0,0),1)]*2
    minimum=float(d.qpos[p.root_qadr+2]); completed=[];finite=True;failed=None
    start=time.monotonic()
    samples=[]
    errors=[];yaw_errors=[];tilts=[];joint_excess=[]
    for command,duration in trace:
        p.set_command(*command)
        begin=d.time
        for step in range(round(duration/model.opt.timestep)):
            p.control(model,d);mujoco.mj_step(model,d)
            height=float(d.qpos[p.root_qadr+2]);minimum=min(minimum,height)
            if step%10==0:
                q=d.qpos[p.root_qadr+3:p.root_qadr+7]
                matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,q)
                local_velocity=matrix.reshape(3,3).T@d.qvel[p.root_dadr:p.root_dadr+3]
                if np.linalg.norm(p.cmd)>=.1 and d.time-begin>.5:
                    errors.append(float(np.sum((local_velocity[:2]-p.cmd[:2])**2)))
                    yaw_errors.append(float((d.qvel[p.root_dadr+5]-p.cmd[2])**2))
                tilts.append(float(np.arccos(np.clip(matrix[8],-1,1))))
                joint_ids=model.actuator_trnid[p.act,0]
                bounds=model.jnt_range[joint_ids]
                positions=d.qpos[p.qadr]
                joint_excess.append(float(np.max(np.maximum(bounds[:,0]-positions,positions-bounds[:,1]))))
            finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(p.action).all())
            if step%50==0:
                samples.append({"t":float(d.time),"position":d.qpos[p.root_qadr:p.root_qadr+3].tolist(),
                     "command":list(command),"action_norm":float(np.linalg.norm(p.action))})
            if not finite or height<.45:
                failed={"reason":"nonfinite" if not finite else "fell_below_0.45m","time":float(d.time),"command":command}
                break
        completed.append({"command":command,"requested_seconds":duration,"actual_seconds":float(d.time-begin)})
        if failed: break
    result={"policy":label,"start_yaw":yaw,
        "linear_velocity_rmse":float(np.sqrt(np.mean(errors))) if errors else None,
        "yaw_velocity_rmse":float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else None,
        "max_tilt_degrees":float(np.degrees(max(tilts))),
        "max_joint_limit_excess_rad":max(0.,max(joint_excess)),"noslip_iterations":noslip,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "loader_accepted":True,"finite":finite,"minimum_root_height":minimum,
        "simulated_seconds":float(d.time),"wall_seconds":time.monotonic()-start,
        "failure":failed,"segments":completed,"samples":samples}
    results.append(result)
    (root/"rollout_result.json").write_text(json.dumps(results,indent=2))
    print("ROLLOUT",{k:v for k,v in result.items() if k not in ("samples","segments")},flush=True)
summary={}
for label in ("supplied","candidate"):
    subset=[x for x in results if x["policy"]==label]
    summary[label]={"passed":sum(x["failure"] is None for x in subset),"trials":len(subset),
        "minimum_survival_seconds":min(x["simulated_seconds"] for x in subset),
        "mean_survival_seconds":float(np.mean([x["simulated_seconds"] for x in subset])),
        "mean_linear_velocity_rmse":float(np.mean([x["linear_velocity_rmse"] for x in subset if x["linear_velocity_rmse"] is not None])),
        "max_tilt_degrees":max(x["max_tilt_degrees"] for x in subset)}
assert summary["supplied"]["passed"]==3,"Supplied baseline failed; investigate test conditions"
(root/"summary.json").write_text(json.dumps(summary,indent=2))
print("DONE",json.dumps(summary),flush=True)
