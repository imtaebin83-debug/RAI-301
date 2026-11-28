"""Isolated HAIC-v2 active-walking training configuration; never edits deployment."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import mujoco
import torch
from contract import *
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlPositionActuatorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

BASE=Path("/home/aril/rai-worktrees/ys-all-v2-20261127")
ASSET=BASE/"src/environment/robots/mujoco/g1/models/g1_dex_sensors_missions.xml"

def robot_spec():
    spec=mujoco.MjSpec.from_file(str(ASSET))
    # Absolute resources survive attaching the source robot to MJLab's world.
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file=str(ASSET.parent/"assets"/Path(mesh.file).name)
    for texture in spec.textures:
        if texture.file:
            texture.file=str(ASSET.parent/texture.file)
    spec.meshdir=""
    for key in list(spec.keys):
        spec.delete(key)
    for name,kp,kd in zip(JOINTS,KP,KD):
        actuator=next(a for a in spec.actuators if a.target==name)
        actuator.gainprm[0]=kp
        actuator.biasprm[1]=-kp
        actuator.biasprm[2]=-kd
    return spec

class CompetitionCommand(UniformVelocityCommand):
    """Slew, phase and restart memory match active calls of the official loader."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.target=torch.zeros_like(self.vel_command_b)
        self.phase=torch.zeros(self.num_envs,device=self.device,dtype=torch.float64)
        self.idle=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.woke=torch.zeros_like(self.idle)
        self.tick=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)

    def _resample_command(self,ids):
        old=self.vel_command_b[ids].clone()
        super()._resample_command(ids)
        self.target[ids]=self.vel_command_b[ids]
        self.target[ids[self.is_standing_env[ids]]]=0
        self.vel_command_b[ids]=old

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids]=0
        self.phase[ids]=0
        self.idle[ids]=True
        self.tick[ids]=-1
        return super().reset(env_ids)

    def _update_command(self):
        self.advance()

    def advance(self):
        fresh=self.tick!=self._env.common_step_counter
        if not fresh.any(): return
        active=torch.linalg.vector_norm(self.target,dim=1)>1e-3
        self.woke=fresh & self.idle & active
        self.phase[self.woke]=0
        self.idle[self.woke]=False
        limit=torch.tensor((1.2,1.2,2.4),device=self.device)*POLICY_DT
        delta=torch.clamp(self.target-self.vel_command_b,-limit,limit)
        self.vel_command_b[fresh]+=delta[fresh]
        stopped=(~active)&(torch.linalg.vector_norm(self.vel_command_b,dim=1)<=1e-3)
        self.idle[fresh & stopped]=True
        self.vel_command_b[fresh & stopped]=0
        running=fresh & ~self.idle
        self.phase[running]=(self.phase[running]+POLICY_DT/GAIT_PERIOD)%1
        self.tick[fresh]=self._env.common_step_counter

@dataclass(kw_only=True)
class CompetitionCommandCfg(UniformVelocityCommandCfg):
    def build(self,env): return CompetitionCommand(self,env)

def pack_observation(ang,gravity,cmd,phase,q,dq,previous):
    moving=torch.linalg.vector_norm(cmd,dim=1)>=0.1
    phase_obs=torch.stack((torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase)),dim=1).to(q.dtype)
    phase_obs[~moving]=0
    default=torch.as_tensor(DEFAULT_POS,device=q.device,dtype=q.dtype)
    return torch.cat((ang,gravity,cmd,phase_obs,q-default,dq,previous),dim=1)

def actor_obs(env):
    command=env.command_manager.get_term("twist")
    command.advance()
    data=env.scene["robot"].data
    previous=env.action_manager.action.clone()
    previous[command.woke | command.idle]=0
    return pack_observation(data.root_link_ang_vel_b,data.projected_gravity_b,
        command.command,command.phase,data.joint_pos,data.joint_vel,previous)

def quat_difference(target,current):
    # MuJoCo mju_subQuat(target,current): log(conjugate(current)*target).
    aw,ax,ay,az=target.unbind(-1)
    bw,bx,by,bz=current.unbind(-1)
    error=torch.stack((bw*aw+bx*ax+by*ay+bz*az,
        bw*ax-bx*aw-by*az+bz*ay,
        bw*ay+bx*az-by*aw-bz*ax,
        bw*az-bx*ay+by*ax-bz*aw),dim=-1)
    error=torch.where(error[:,:1]<0,-error,error)
    norm=torch.linalg.vector_norm(error[:,1:],dim=-1,keepdim=True)
    angle=2*torch.atan2(norm,error[:,:1])
    return error[:,1:]*(angle/norm.clamp_min(1e-12))

def official_idle_wrench(pos,quat,velocity,held_pos,held_quat,idle):
    force=(torch.as_tensor((6000.,6000.,2500.),device=pos.device)*(held_pos-pos)
        -torch.as_tensor((300.,300.,220.),device=pos.device)*velocity[:,:3]).clamp(-5000,5000)
    torque=(1000*quat_difference(held_quat,quat)-80*velocity[:,3:6]).clamp(-1000,1000)
    return torch.cat((force,torque),dim=1)*idle[:,None]

class CompetitionAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.holding=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.held_pos=torch.zeros(env.num_envs,3,device=env.device)
        self.held_quat=torch.zeros(env.num_envs,4,device=env.device)
        self.held_quat[:,0]=1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.holding[slice(None) if env_ids is None else env_ids]=False

    def apply_actions(self):
        command=self._env.command_manager.get_term("twist")
        target=self._processed_actions.clone()
        target[command.idle]=self._offset[command.idle]
        self._entity.set_joint_position_target(target,joint_ids=self._target_ids)
        qids=self._entity.indexing.free_joint_q_adr
        vids=self._entity.indexing.free_joint_v_adr
        q=self._env.sim.data.qpos[:,qids]
        velocity=self._env.sim.data.qvel[:,vids]
        capture=command.idle & ~self.holding
        self.held_pos[capture]=q[capture,:3]
        self.held_quat[capture]=q[capture,3:7]
        self.holding[:]=command.idle
        self._env.sim.data.qfrc_applied[:,vids]=official_idle_wrench(
            q[:,:3],q[:,3:7],velocity,self.held_pos,self.held_quat,command.idle)

@dataclass(kw_only=True)
class CompetitionActionCfg(JointPositionActionCfg):
    def build(self,env): return CompetitionAction(self,env)

def sample_curriculum_targets(n,stage,device):
    assert stage in (0,1,2)
    u=torch.rand(n,device=device)
    target=torch.zeros(n,3,device=device)
    # Keep both new directions present even when the curriculum steps back.
    limits=[(.10,.12),(.10,.18),(.10,.25)]
    low,high=limits[stage]
    forward=u>=.60
    backward=(u>=.30)&(u<.60)
    lateral=(u>=.05)&(u<.30)
    turn=u<.05
    target[forward,0]=torch.empty(int(forward.sum()),device=device).uniform_(.15,.3)
    target[backward,0]=-torch.empty(int(backward.sum()),device=device).uniform_(low,high)
    target[lateral,1]=torch.where(torch.rand(int(lateral.sum()),device=device)<(2/3),-1.,1.)*torch.empty(int(lateral.sum()),device=device).uniform_(low,high)
    target[turn,2]=torch.where(torch.rand(int(turn.sum()),device=device)<.5,-1.,1.)*torch.empty(int(turn.sum()),device=device).uniform_(.1,.15)
    return target

class WalkingCurriculumCommand(CompetitionCommand):
    def reset(self,env_ids=None):
        if hasattr(self,"cycle"):
            self.cycle[slice(None) if env_ids is None else env_ids]=3
        return super().reset(env_ids)

    def _resample_command(self,ids):
        super()._resample_command(ids)
        if not hasattr(self,"cycle"):
            self.cycle=torch.full((self.num_envs,),3,device=self.device,dtype=torch.long)
            self.saved_command=torch.zeros_like(self.target)
            self.heading_pending=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
            self.heading_reference=torch.zeros(self.num_envs,device=self.device)
        phases=self.cycle[ids]%4
        new_ids=ids[phases==0]
        if len(new_ids):
            self.saved_command[new_ids]=sample_curriculum_targets(len(new_ids),getattr(self,"stage",0),self.device)
        moving=(phases==0)|(phases==2)
        self.target[ids]=torch.where(moving[:,None],self.saved_command[ids],torch.zeros_like(self.target[ids]))
        # Jitter gait phase at stopping; practice a range of pauses and the same restart.
        duration=torch.empty(len(ids),device=self.device)
        duration[moving]=torch.empty(int(moving.sum()),device=self.device).uniform_(3.5,4.5)
        duration[~moving]=torch.empty(int((~moving).sum()),device=self.device).uniform_(1.,2.)
        self.time_left[ids]=duration
        # Keep one heading reference through move-stop-restart; do not forgive
        # accumulated rotation at the pause or the second movement burst.
        self.heading_pending[new_ids]=True
        self.cycle[ids]+=1
        self.is_standing_env[ids]=torch.linalg.vector_norm(self.target[ids],dim=1)<.001

@dataclass(kw_only=True)
class WalkingCurriculumCommandCfg(CompetitionCommandCfg):
    def build(self,env): return WalkingCurriculumCommand(self,env)

def stability_cost_values(command,velocity,angular_velocity,height):
    # Dense bounded tracking costs preserve a signal when exponential rewards saturate.
    xy=((velocity[:,:2]-command[:,:2])**2).sum(1).div(.04).clamp(max=4)
    yaw=((angular_velocity[:,2]-command[:,2])**2).clamp(max=4)
    support=torch.exp(-((height-.78)/.06)**2)
    return xy,yaw,support

def action_magnitude(env):
    return torch.sum(env.action_manager.action**2,dim=1)*(~env.command_manager.get_term("twist").idle)

def stability_term(env,component):
    data=env.scene["robot"].data
    return stability_cost_values(env.command_manager.get_command("twist"),
        data.root_link_lin_vel_b,data.root_link_ang_vel_b,
        data.root_link_pos_w[:,2]-env.scene.env_origins[:,2])[component]

def directional_progress_score(command,velocity,gravity,height):
    # Bounded reward for moving in the requested direction. Falling and excess speed
    # cannot increase it without bound. Zero commands get zero progress reward.
    norm_sq=torch.sum(command[:,:2]**2,dim=1)
    projection=torch.sum(command[:,:2]*velocity[:,:2],dim=1)/norm_sq.clamp_min(.01)
    upright=(-gravity[:,2]).clamp(0,1)**2
    return projection.clamp(-1,1)*(norm_sq>=.01)*upright*(height>.6)

def commanded_progress_error(command,velocity):
    norm=torch.linalg.vector_norm(command[:,:2],dim=1)
    unit=command[:,:2]/norm[:,None].clamp_min(.1)
    speed=(unit*velocity[:,:2]).sum(1)
    # Relative error matters equally for slow backward/sideways and faster forward.
    # Zero, ramping and turning commands are excluded; excess speed is not rewarded.
    active=(norm>=.1)&(command[:,2].abs()<=.001)
    return ((speed-norm)/norm.clamp_min(.1)).square().clamp(max=4)*active

def progress_error(env):
    data=env.scene["robot"].data
    return commanded_progress_error(env.command_manager.get_command("twist"),data.root_link_lin_vel_b)

def directional_progress(env):
    robot=env.scene["robot"].data
    command=env.command_manager.get_command("twist")
    score=directional_progress_score(command,robot.root_link_lin_vel_b,
        robot.projected_gravity_b,robot.root_link_pos_w[:,2]-env.scene.env_origins[:,2])
    env.extras["log"]["Metrics/command_stage"]=float(getattr(env.command_manager.get_term("twist"),"stage",0))
    env.extras["log"]["Metrics/directional_progress"]=score.mean()
    return score

def heading_drift_cost(current,reference,command):
    error=torch.atan2(torch.sin(current-reference),torch.cos(current-reference))
    lateral=(torch.linalg.vector_norm(command[:,:2],dim=1)>=.05)&(command[:,2].abs()<=.001)
    # Smooth bounded cost keeps distinguishing 60-degree and 120-degree drift.
    return 2*(1-torch.cos(error))*lateral

def sideways_heading(env):
    command=env.command_manager.get_term("twist")
    data=env.scene["robot"].data
    yaw=data.heading_w
    pending=command.heading_pending
    command.heading_reference[pending]=yaw[pending]
    command.heading_pending[:]=False
    return heading_drift_cost(yaw,command.heading_reference,command.command)

def make_cfg(num_envs=512):
    cfg=unitree_g1_flat_env_cfg()
    cfg.scene.num_envs=num_envs
    cfg.scene.entities={"robot":EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,0.79),
            joint_pos=dict(zip(JOINTS,map(float,DEFAULT_POS))),joint_vel={".*":0.0}),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlPositionActuatorCfg(target_names_expr=(".*",)),)),
    )}
    cfg.actions["joint_pos"]=CompetitionActionCfg(entity_name="robot",
        actuator_names=(".*",),scale=dict(zip(JOINTS,map(float,ACTION_SCALE))))
    cfg.commands["twist"]=WalkingCurriculumCommandCfg(entity_name="robot",
        resampling_time_range=(4.0,7.0),rel_standing_envs=0.1,
        heading_command=False,rel_heading_envs=0.0,init_velocity_prob=0.0,debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5,1.0),lin_vel_y=(-0.5,0.5),ang_vel_z=(-1.0,1.0)))
    cfg.observations["actor"]=ObservationGroupCfg(
        terms={"competition":ObservationTermCfg(func=actor_obs)},
        concatenate_terms=True,enable_corruption=False)
    # Privileged critic is training-only. No linear velocity/contact/height enters actor.
    cfg.observations["critic"].terms["competition"]=ObservationTermCfg(func=actor_obs)
    cfg.decimation=CONTROL_DECIMATION
    cfg.sim.mujoco.timestep=NOMINAL_DT
    cfg.sim.mujoco.gravity=(0,0,-9.8)
    cfg.sim.mujoco.iterations=100
    cfg.sim.mujoco.ls_iterations=50
    cfg.sim.mujoco.jacobian="sparse"
    cfg.sim.mujoco.multiccd=True
    cfg.sim.mujoco.cone="elliptic"
    cfg.sim.mujoco.impratio=10
    cfg.sim.nconmax=128
    cfg.sim.njmax=512
    # Stage 1 checks exact nominal mechanics before robustness randomization.
    cfg.events={k:v for k,v in cfg.events.items() if k in ("reset_base","reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"]={"yaw":(-3.14,3.14)}
    cfg.curriculum={}
    cfg.rewards["track_linear_velocity"].weight=5.0
    cfg.rewards["track_linear_velocity"].params["std"]=.18
    cfg.rewards["track_angular_velocity"].weight=4.0
    cfg.rewards["track_angular_velocity"].params["std"]=.25
    cfg.rewards["pose"].weight=.2
    cfg.rewards["action_rate_l2"].weight=-.04
    cfg.rewards["directional_progress"]=RewardTermCfg(func=directional_progress,weight=1.0)
    cfg.rewards["upright"].weight=2.0
    cfg.rewards["upright"].params["std"]=.316227766
    cfg.rewards["body_ang_vel"].weight=-.1
    cfg.rewards["dof_pos_limits"].weight=-3.0
    cfg.rewards["action_magnitude"]=RewardTermCfg(func=action_magnitude,weight=-.002)
    cfg.rewards["dense_velocity_error"]=RewardTermCfg(func=stability_term,weight=-.4,params={"component":0})
    cfg.rewards["unwanted_yaw"]=RewardTermCfg(func=stability_term,weight=-1.0,params={"component":1})
    cfg.rewards["height_support"]=RewardTermCfg(func=stability_term,weight=.5,params={"component":2})
    # All regularization changes are training-only; the deployed action mapping is unchanged.
    cfg.rewards["sideways_heading"]=RewardTermCfg(func=sideways_heading,weight=-1.0)
    cfg.rewards["commanded_progress_error"]=RewardTermCfg(func=progress_error,weight=-.8)
    return cfg
