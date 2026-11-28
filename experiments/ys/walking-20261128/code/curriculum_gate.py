import math
def forward_gate(trials):
    candidates=[x for x in trials if x["policy"]=="candidate"]
    passed=bool(candidates)
    details=[]
    for row in candidates:
        group=[]
        for sample in row["samples"]:
            if sample["command"]==[.25,0,0]: group.append(sample)
            elif group: break
        forward=side=0.
        if len(group)>1:
            dx=group[-1]["position"][0]-group[0]["position"][0]
            dy=group[-1]["position"][1]-group[0]["position"][1]
            yaw=row["start_yaw"]
            forward=dx*math.cos(yaw)+dy*math.sin(yaw)
            side=-dx*math.sin(yaw)+dy*math.cos(yaw)
        ok=(row["failure"] is None and forward>=.6 and abs(side)<=.25
            and row["linear_velocity_rmse"] is not None and row["linear_velocity_rmse"]<=.15
            and row["yaw_velocity_rmse"] is not None and row["yaw_velocity_rmse"]<=.25
            and row["max_tilt_degrees"]<20)
        details.append({"forward_m":forward,"lateral_drift_m":side,"pass":ok})
        passed=passed and ok
    return passed,details

def next_stage(current,forward_ok,broad_ok,skills_ok=False):
    if not forward_ok:return max(0,current-1)
    if not skills_ok:return current
    if current==0:return 1
    if current==1 and broad_ok:return 2
    return current

def skill_gate(trials,direction):
    rows=[x for x in trials if x["policy"]=="candidate"]
    details=[]
    for row in rows:
        segments=row.get("segments",[])
        active=[s for s in segments if s["command"]==list(direction)]
        stops=[s for s in segments if s["command"]==[0,0,0]]
        distances=[s.get("distance_m",0.) for s in active]
        drifts=[abs(s.get("perpendicular_drift_m",float("inf"))) for s in active]
        heading=max((s.get("max_heading_error_rad",float("inf")) for s in segments),default=float("inf"))
        stop_drift=max((s.get("displacement_m",float("inf")) for s in stops),default=float("inf"))
        ok=(row["failure"] is None and len(active)==2
            and all(s["actual_seconds"]>=3.99 for s in active)
            and min(distances,default=0.)>=.2 and max(drifts,default=float("inf"))<=.2
            and heading<=.35 and stop_drift<=.10
            and row["linear_velocity_rmse"] is not None and row["linear_velocity_rmse"]<=.14
            and row["yaw_velocity_rmse"] is not None and row["yaw_velocity_rmse"]<=.25
            and row["max_tilt_degrees"]<20)
        details.append({"start_yaw":row.get("start_yaw"),"burst_distances_m":distances,
            "burst_perpendicular_drift_m":drifts,"max_heading_error_rad":heading,
            "max_stop_drift_m":stop_drift,"pass":ok})
    headings=sorted(x["start_yaw"] for x in details if isinstance(x["start_yaw"],(int,float)))
    orientations_ok=len(headings)==3 and all(abs(a-b)<1e-6 for a,b in zip(headings,(-.2,0.,.2)))
    return orientations_ok and all(x["pass"] for x in details),details
