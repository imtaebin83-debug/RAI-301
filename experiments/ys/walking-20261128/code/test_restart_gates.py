import copy,json
from pathlib import Path
from curriculum_gate import skill_gate
direction=(-.12,0,0)
move={"command":list(direction),"actual_seconds":4.,"distance_m":.3,"perpendicular_drift_m":.02,"max_heading_error_rad":.1}
stop={"command":[0,0,0],"actual_seconds":1.,"displacement_m":.02,"max_heading_error_rad":.1}
row={"policy":"candidate","failure":None,"linear_velocity_rmse":.05,"yaw_velocity_rmse":.1,"max_tilt_degrees":5,
 "segments":[stop,move,stop,stop,move,stop]}
rows=[{**copy.deepcopy(row),"start_yaw":yaw} for yaw in (-.2,0.,.2)]
assert skill_gate(rows,direction)[0]
for field,value,index in [("distance_m",.01,4),("distance_m",-.3,4),("max_heading_error_rad",.4,4),("displacement_m",.11,2)]:
 bad=copy.deepcopy(rows);bad[1]["segments"][index][field]=value
 assert not skill_gate(bad,direction)[0],(field,value)
assert not skill_gate(rows[:1],direction)[0]
assert not skill_gate([rows[0]]*3,direction)[0]
Path("restart_gate_test_result.json").write_text(json.dumps({"both_bursts_required":True,"wrongway_second_burst_rejected":True,"heading_and_stop_drift_checked":True,"three_orientations_required":True},indent=2))
print("Restart/heading gate tests passed")
