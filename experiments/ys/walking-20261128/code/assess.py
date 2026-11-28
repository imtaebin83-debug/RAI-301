"""Readiness checks use exported-policy motion, not training reward."""
from pathlib import Path
import json,math
root=Path(__file__).resolve().parent
def assess():
    rows=[]
    for folder in sorted(root.glob("iteration_*")):
        if not (folder/"summary.json").exists(): continue
        short=json.loads((folder/"summary.json").read_text())
        trials=json.loads((folder/"rollout_result.json").read_text())
        displacements=[]
        ratios=[]
        for row in trials:
            if row["policy"]!="candidate": continue
            forward=[]
            for sample in row["samples"]:
                if sample["command"]==[.25,0,0]:
                    if forward and sample["t"]-forward[-1]["t"]>.15: break
                    forward.append(sample)
                elif forward: break
            if len(forward)<2:
                displacements.append(0.);ratios.append(0.);continue
            dx=forward[-1]["position"][0]-forward[0]["position"][0]
            dy=forward[-1]["position"][1]-forward[0]["position"][1]
            distance=dx*math.cos(row["start_yaw"])+dy*math.sin(row["start_yaw"])
            displacements.append(distance)
            ratios.append(distance/(.25*(forward[-1]["t"]-forward[0]["t"])))
        extended_file=folder/"extended/summary.json"
        extended=json.loads(extended_file.read_text()) if extended_file.exists() else None
        gate_file=folder/"curriculum_gate.json"
        gate=json.loads(gate_file.read_text()) if gate_file.exists() else {}
        skills=gate.get("skills",{})
        directions_ok=len(skills)==3 and all(x["pass"] for x in skills.values())
        candidate=short["candidate"]
        useful_motion=bool(displacements) and min(displacements)>=.5 and min(ratios)>=.5
        short_ok=(candidate["passed"]==3 and useful_motion
            and candidate["mean_linear_velocity_rmse"]<=.18 and candidate["max_tilt_degrees"]<30)
        extended_ok=extended is not None and extended["candidate"]["passed"]==1 and extended["candidate"]["mean_linear_velocity_rmse"]<=.18 and extended["candidate"]["max_tilt_degrees"]<30
        rows.append({"iteration":int(folder.name.split("_")[1]),
            "short_summary":short,"extended_summary":extended,
            "forward_displacement_m":displacements,"forward_requested_distance_ratio":ratios,
            "useful_commanded_motion":useful_motion,"short_gate":short_ok,
            "extended_gate":extended_ok,"direction_gate":directions_ok,"all_motion_gates":short_ok and extended_ok and directions_ok})
    extended_rows=[r for r in rows if r["extended_summary"] is not None]
    ready=len(extended_rows)>=2 and all(r["all_motion_gates"] for r in extended_rows[-2:])
    result={"ready_for_overnight_training":ready,"overnight_started":False,
        "scope":"Evidence for longer learning; does not authorize policy deployment",
        "interpretation":"Requires commanded movement and repeatable short/long official-physics tests at two later checkpoints",
        "criteria":{"minimum_forward_displacement_m":.5,"minimum_forward_requested_distance_ratio":.5,
            "maximum_linear_velocity_rmse":.18,"maximum_tilt_degrees":30,
            "short_trials_without_fall":3,"extended_trials_without_fall":1},
        "checkpoints":rows}
    (root/"assessment.json").write_text(json.dumps(result,indent=2))
    return result
if __name__=="__main__":
    print(json.dumps(assess(),indent=2))
