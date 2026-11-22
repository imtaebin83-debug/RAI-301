"""Re-run after benchmark cases finish; reads evaluation artifacts only.

Does not import policy/simulator modules or modify custom/ source. All statistics
exclude a still-running case until its atomic result.json has been written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def write_json(path,value):
    temporary=path.with_name(path.name+"."+str(os.getpid())+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def stat(values):
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"count":len(values),"mean":statistics.mean(values) if values else None,
            "median":statistics.median(values) if values else None,
            "min":min(values) if values else None,"max":max(values) if values else None,
            "values":values}


def load_run(folder):
    path=folder/"run_manifest.json"
    manifest=json.loads(path.read_text()) if path.is_file() else {}
    rows={}
    for result in sorted(folder.glob("case_*/result.json")):
        row=json.loads(result.read_text())
        index=int(row["index"])
        if index in rows:
            raise ValueError("Duplicate case index "+str(index)+" in "+str(folder))
        row["_directory"]=result.parent
        row["_scenario_verified"]=(result.parent/"scenario.yaml").is_file() and digest(result.parent/"scenario.yaml")==row.get("scenario_sha256")
        before=row.get("artifacts_before",{})
        row["_code_sha256"]=canonical_hash(before.get("files",{}))
        row["_model_sha256"]=before.get("checkpoint",{}).get("sha256")
        rows[index]=row
    requested=manifest.get("indices",[])
    if not requested and manifest:
        requested=list(range(manifest.get("cases",0)))
    return {"folder":folder,"manifest":manifest,"rows":rows,"requested":requested}


def seconds(row):
    return (row.get("official_record") or {}).get("seconds")


def with_expectation(run,seed,indices):
    """Predeclare an evaluation in analysis without writing a harness manifest."""
    run["expected_seed"]=seed;run["expected_indices"]=list(indices)
    run["manifest_matches_declared_plan"]=(run["manifest"].get("seed")==seed
        and set(run["requested"])==set(indices)) if run["manifest"] else None
    if not run["manifest"]:
        run["manifest"]={"seed":seed,"indices":list(indices),"cases":len(indices),
                         "analysis_declaration_only":True}
        run["requested"]=list(indices)
    return run


def eligible_rows(run):
    # An interrupted run has no outcome. Preserve it in the audit trail only.
    return {i:row for i,row in run["rows"].items() if not row.get("interrupted")}


def run_summary(run):
    rows=run["rows"]
    eligible=eligible_rows(run)
    passed=[i for i,row in eligible.items() if row.get("success") is True]
    failed=[i for i,row in eligible.items() if row.get("success") is False]
    unjudged=[i for i,row in eligible.items() if row.get("success") is None]
    interrupted=[i for i,row in rows.items() if row.get("interrupted")]
    judged=set(passed+failed)
    codes=sorted({row["_code_sha256"] for row in rows.values()})
    models=sorted({row["_model_sha256"] for row in rows.values() if row["_model_sha256"]})
    requested=run["requested"]
    return {"folder":str(run["folder"]),"suite_seed":run["manifest"].get("seed"),
            "requested":len(requested),"completed_attempts":len(rows),
            "performance_attempts":len(judged),"noninterrupted_attempts":len(eligible),
            "interrupted_attempts":len(interrupted),"interrupted_indices":interrupted,
            "pending_indices":[i for i in requested if i not in rows],
            "pending_official_evaluation_indices":[i for i in requested if i not in judged],
            "passed":len(passed),"failed":len(failed),"not_judged":len(unjudged),
            "pass_indices":passed,"fail_indices":failed,"unjudged_indices":unjudged,
            "pass_fraction_completed":len(passed)/len(judged) if judged else None,
            "pass_fraction_denominator":"officially judged, noninterrupted attempts",
            "official_seconds_all_completed":stat([seconds(row) for row in eligible.values()]),
            "official_seconds_success_only":stat([seconds(rows[i]) for i in passed]),
            "scenario_files_match_recorded_sha":all(row["_scenario_verified"] for row in rows.values()),
            "code_sha256_values":codes,"model_sha256_values":models,
            "frozen_code_across_completed_cases":len(codes)<=1,
            "frozen_model_across_completed_cases":len(models)<=1,
            "changed_during_case_indices":[i for i,row in rows.items() if row.get("artifacts_changed_during_case")],
            "sampler_sha256":run["manifest"].get("sampler_sha256"),
            "run_manifest_present":bool(run["manifest"]) and not run["manifest"].get("analysis_declaration_only",False),
            "declared_expected_seed":run.get("expected_seed"),
            "declared_expected_indices":run.get("expected_indices"),
            "manifest_matches_declared_plan":run.get("manifest_matches_declared_plan"),
            "evaluation_complete":bool(requested) and set(requested)==judged,
            "interrupted_note":"Interrupted attempts are preserved as audit records and excluded from outcome/timing denominators."}


def compare(baseline,candidate):
    b,c=eligible_rows(baseline),eligible_rows(candidate)
    common=sorted(set(b)&set(c))
    pairing=[]
    for i in common:
        x,y=b[i],c[i]
        pairing.append({"index":i,"baseline_seed":x.get("case_seed"),"candidate_seed":y.get("case_seed"),
                        "same_case_seed":x.get("case_seed")==y.get("case_seed"),
                        "same_suite_seed":x.get("suite_seed")==y.get("suite_seed"),
                        "scenario_sha256":x.get("scenario_sha256"),
                        "same_scenario_sha256":x.get("scenario_sha256")==y.get("scenario_sha256"),
                        "baseline_file_sha_verified":x["_scenario_verified"],
                        "candidate_file_sha_verified":y["_scenario_verified"],
                        "baseline_success":x.get("success"),"candidate_success":y.get("success"),
                        "baseline_official_seconds":seconds(x),"candidate_official_seconds":seconds(y),
                        "delta_official_seconds_candidate_minus_baseline":
                            seconds(y)-seconds(x) if seconds(x) is not None and seconds(y) is not None else None,
                        "baseline_failure_stage":x.get("failure_stage"),
                        "candidate_failure_stage":y.get("failure_stage")})
    excluded_interrupted=[{"index":i,"baseline_interrupted":bool(baseline["rows"][i].get("interrupted")),
                           "candidate_interrupted":bool(candidate["rows"][i].get("interrupted"))}
                          for i in sorted(set(baseline["rows"])&set(candidate["rows"]))
                          if baseline["rows"][i].get("interrupted") or candidate["rows"][i].get("interrupted")]
    shared=[i for i in common if b[i].get("success") is True and c[i].get("success") is True]
    gained=[i for i in common if b[i].get("success") is False and c[i].get("success") is True]
    lost=[i for i in common if b[i].get("success") is True and c[i].get("success") is False]
    bs,cs=run_summary(baseline),run_summary(candidate)
    sampler_same=bs["sampler_sha256"] is not None and bs["sampler_sha256"]==cs["sampler_sha256"]
    verified=all(row[key] for row in pairing for key in
                 ("same_case_seed","same_suite_seed","same_scenario_sha256","baseline_file_sha_verified","candidate_file_sha_verified"))
    frozen=all(summary["frozen_code_across_completed_cases"] and summary["frozen_model_across_completed_cases"]
               and not summary["changed_during_case_indices"] for summary in (bs,cs))
    requested_match=bool(baseline["requested"]) and set(baseline["requested"])==set(candidate["requested"])
    plan_valid=all(summary["manifest_matches_declared_plan"] is not False for summary in (bs,cs))
    complete=requested_match and all(summary["evaluation_complete"] and summary["run_manifest_present"] for summary in (bs,cs))
    b_times=[seconds(b[i]) for i in shared];c_times=[seconds(c[i]) for i in shared]
    return {"baseline":bs,"candidate":cs,"pair_count":len(common),
            "unpaired_baseline_indices":sorted(set(b)-set(c)),
            "unpaired_candidate_indices":sorted(set(c)-set(b)),
            "same_official_sampler_source":sampler_same,
            "all_completed_pairs_verified":verified,
            "valid_for_frozen_variant_comparison":bool(common) and verified and sampler_same and frozen and plan_valid,
            "requested_indices_match":requested_match,"comparison_complete":complete,
            "valid_for_full_requested_comparison":complete and verified and sampler_same and frozen and plan_valid,
            "excluded_interrupted_pairs":excluded_interrupted,
            "completed_paired_outcomes":{
                side:{"passed":sum(rows[i].get("success") is True for i in common),
                      "failed":sum(rows[i].get("success") is False for i in common),
                      "not_judged":sum(rows[i].get("success") is None for i in common),
                      "official_seconds_all":stat([seconds(rows[i]) for i in common])}
                for side,rows in (("baseline",b),("candidate",c))},
            "gained_pass_indices":gained,"lost_pass_indices":lost,
            "shared_pass_indices":shared,
            "official_seconds_shared_pass_only":{"baseline":stat(b_times),"candidate":stat(c_times),
                "paired_delta_candidate_minus_baseline":stat([y-x for x,y in zip(b_times,c_times)
                                                             if x is not None and y is not None])},
            "pairs":pairing,
            "notes":["Official seconds include failures in the all-completed statistic; faster early failure is not an improvement.",
                     "Shared-PASS timing compares only cases passed by both variants.",
                     "Interrupted attempts are excluded from performance pairs and kept separately; null success is never a failure.",
                     "Scenario replay fixes placement, not bitwise ROS scheduling or physics trajectories."]}


def combine_suites(suites):
    """Aggregate every observed pair, with suite seed retained in each case key."""
    pairs=[];requested=set();variant_runs={"baseline":[],"candidate":[]}
    for label,b,c,comparison in suites:
        for run in (b,c):
            seed=run["manifest"].get("seed")
            requested.update((seed,index) for index in run["requested"])
        variant_runs["baseline"].append(b);variant_runs["candidate"].append(c)
        for row in comparison["pairs"]:
            pairs.append({"suite":label,"suite_seed":c["manifest"].get("seed"),**row})
    keys=[(row["suite_seed"],row["index"]) for row in pairs]
    if len(keys)!=len(set(keys)):
        raise ValueError("Duplicate suite seed/index pairs in combined comparison")
    shared=[row for row in pairs if row["baseline_success"] is True and row["candidate_success"] is True]
    key=lambda row:{"suite":row["suite"],"suite_seed":row["suite_seed"],"index":row["index"]}
    outcomes={};fingerprints={};all_scenarios=set();all_placements=set();samplers=set()
    for side,runs in variant_runs.items():
        codes=set();models=set();completed=0;performance=0;interrupted=0;changed=[]
        for run in runs:
            summary=run_summary(run);completed+=summary["completed_attempts"]
            performance+=summary["performance_attempts"];interrupted+=summary["interrupted_attempts"]
            if summary["sampler_sha256"]:
                samplers.add(summary["sampler_sha256"])
            codes.update(summary["code_sha256_values"]);models.update(summary["model_sha256_values"])
            changed.extend({"suite_seed":summary["suite_seed"],"index":i} for i in summary["changed_during_case_indices"])
            for row in run["rows"].values():
                all_scenarios.add(row.get("scenario_sha256"))
                scenario=yaml.safe_load((row["_directory"]/"scenario.yaml").read_text())
                all_placements.add(canonical_hash(scenario["panda_task"]))
        outcomes[side]={"completed_physical_runs":completed,"performance_physical_runs":performance,
            "interrupted_physical_runs":interrupted,"completed_pairs":len(pairs),
            "passed_in_completed_pairs":sum(row[side+"_success"] is True for row in pairs),
            "failed_in_completed_pairs":sum(row[side+"_success"] is False for row in pairs),
            "not_judged_in_completed_pairs":sum(row[side+"_success"] is None for row in pairs),
            "pass_fraction_completed_pairs":sum(row[side+"_success"] is True for row in pairs)/len(pairs) if pairs else None,
            "official_seconds_all_completed_pairs":stat(row[side+"_official_seconds"] for row in pairs)}
        fingerprints[side]={"code_sha256_values":sorted(codes),"model_sha256_values":sorted(models),
            "frozen_code_across_suites":len(codes)<=1,"frozen_model_across_suites":len(models)<=1,
            "changed_during_case":changed}
    complete=all(comparison["comparison_complete"] for _,_,_,comparison in suites)
    valid=all(comparison["valid_for_full_requested_comparison"] for _,_,_,comparison in suites)
    frozen=all(fp["frozen_code_across_suites"] and fp["frozen_model_across_suites"] and not fp["changed_during_case"]
               for fp in fingerprints.values())
    return {"suite_labels":[label for label,_,_,_ in suites],
        "requested_unique_case_keys":len(requested),"planned_physical_runs":2*len(requested),
        "completed_physical_runs":sum(v["completed_physical_runs"] for v in outcomes.values()),
        "performance_physical_runs":sum(v["performance_physical_runs"] for v in outcomes.values()),
        "interrupted_physical_runs":sum(v["interrupted_physical_runs"] for v in outcomes.values()),
        "completed_pairs":len(pairs),"pending_pairs":len(requested)-len(pairs),
        "observed_unique_scenario_sha256_count":len(all_scenarios-{None}),
        "observed_unique_panda_task_placement_count":len(all_placements),
        "sampler_sha256_values":sorted(samplers),"same_official_sampler_source_across_suites":len(samplers)==1,
        "all_requested_pairs_complete":complete,"valid_for_full_frozen_comparison":valid and frozen and len(samplers)==1,
        "variant_fingerprints":fingerprints,"outcomes_on_completed_pairs":outcomes,
        "gained_pass_cases":[key(row) for row in pairs if row["baseline_success"] is False and row["candidate_success"] is True],
        "lost_pass_cases":[key(row) for row in pairs if row["baseline_success"] is True and row["candidate_success"] is False],
        "shared_pass_cases":[key(row) for row in shared],
        "official_seconds_shared_pass_only":{
            "baseline":stat(row["baseline_official_seconds"] for row in shared),
            "candidate":stat(row["candidate_official_seconds"] for row in shared),
            "paired_delta_candidate_minus_baseline":stat(row["delta_official_seconds_candidate_minus_baseline"] for row in shared)},
        "pairs":pairs,
        "notes":["Development cases and held-out cases remain separate suites; the combined result is descriptive, not an independent held-out estimate.",
                 "Every noninterrupted completed pair is included, including both variants' failures. Pending cases are not successes.",
                 "Interrupted runs remain in physical-run audit counts but are excluded from performance pairs and timing.",
                 "Outcome rates use only completed pairs, so a faster variant cannot gain a larger denominator during an ongoing comparison.",
                 "All-trial seconds include early failures; use shared-PASS seconds for comparable completion-time measurements.",
                 "Placement uniqueness hashes are calculated only from completed evaluation scenario panda_task fields, never policy inputs."]}


def coverage(run):
    metrics={key:[] for key in ("pencil_radius_m","cup_radius_m","pencil_angle_deg","cup_angle_deg",
                                "pencil_yaw_deg","cup_yaw_deg","object_separation_m",
                                "pencil_world_x","pencil_world_y","cup_world_x","cup_world_y")}
    cases=[]
    for index,row in run["rows"].items():
        scenario=yaml.safe_load((row["_directory"]/"scenario.yaml").read_text())
        task=scenario["panda_task"]
        base=row.get("panda_base_xy")
        if not base or len(base)!=2:
            raise ValueError("Missing sampled base XY")
        p,c=task["pencil_xy"],task["cup_xy"]
        def polar(point):
            x,y=point[0]-base[0],point[1]-base[1]
            return math.hypot(x,y),math.degrees(math.atan2(y,x))%360.
        pr,pa=polar(p);cr,ca=polar(c)
        values={"pencil_radius_m":pr,"cup_radius_m":cr,"pencil_angle_deg":pa,"cup_angle_deg":ca,
                "pencil_yaw_deg":task["pencil_yaw_deg"],"cup_yaw_deg":task["cup_yaw_deg"],
                "object_separation_m":math.hypot(p[0]-c[0],p[1]-c[1]),
                "pencil_world_x":p[0],"pencil_world_y":p[1],"cup_world_x":c[0],"cup_world_y":c[1]}
        for key,value in values.items():metrics[key].append(float(value))
        cases.append({"index":index,"case_seed":row.get("case_seed"),"success":row.get("success"),
                      "interrupted":bool(row.get("interrupted")),
                      "scenario_sha256":row.get("scenario_sha256"),**values})
    return {"suite_seed":run["manifest"].get("seed"),"completed_attempts":len(cases),
            "pending_indices":run_summary(run)["pending_indices"],
            "observed_ranges":{key:{"min":min(values) if values else None,"max":max(values) if values else None}
                               for key,values in metrics.items()},
            "pencil_outer_radius_over_0_60m_count":sum(v>.60 for v in metrics["pencil_radius_m"]),
            "room_orders":sorted({tuple(yaml.safe_load((row["_directory"]/"scenario.yaml").read_text())["room_order"])
                                  for row in run["rows"].values()}),
            "cases":cases,
            "notes":["All coordinates are from evaluation scenario specifications, not policy inputs or sensor estimates.",
                     "Ranges describe sampled completed cases only, not complete coverage of the official distribution."]}


def progress(run,development):
    current=run_summary(run);reference=run_summary(development)
    old_hashes={row.get("scenario_sha256") for row in development["rows"].values()}
    hashes={row.get("scenario_sha256") for row in run["rows"].values()}
    return {**current,"development_seed":development["manifest"].get("seed"),
        "different_suite_seed_from_development":run["manifest"].get("seed")!=development["manifest"].get("seed"),
        "development_scenario_hash_overlap":sorted((old_hashes&hashes)-{None}),
        "same_frozen_code_as_candidate":current["code_sha256_values"]==reference["code_sha256_values"] if run["rows"] else None,
        "same_frozen_model_as_candidate":current["model_sha256_values"]==reference["model_sha256_values"] if run["rows"] else None,
        "note":"Only atomic result.json files are read. Pending/interrupted trials are not successes; interrupted trials do not enter performance denominators."}


def version_plan(out,development_seed,development_indices,holdout_seed,holdout_cases,generated,version="v2",inherited=None):
    expected={"development_seed":development_seed,"development_indices":list(development_indices),
              "held_out_seed":holdout_seed,"held_out_indices":list(range(holdout_cases)),
              "held_out_cases":holdout_cases}
    path=out/(version+"_evaluation_plan.json")
    if path.is_file():
        existing=json.loads(path.read_text())
        if existing.get("predeclared")!=expected:
            raise ValueError(version+" evaluation differs from the previously saved predeclared plan")
        return existing
    plan={"version":version,"declared_at_kst":generated,"predeclared":expected,
          "protocol":"Development baseline paired comparison, then a separately reported fresh-seed holdout.",
          "candidate_freeze_requirement":"Freeze source and model before running the declared suite; verify artifact hashes afterward.",
          "reason_for_new_holdout":"V1 was rejected after a held-out regression; seed221122 is now observed development evidence.",
          "selection_rule":"Run and report every declared case, including failures. No favorable-seed or successful-case selection.",
          "pairing_rule":"The new-seed holdout cannot be paired with the old-seed baseline. A matching-seed baseline would require a separately declared evaluation."}
    if inherited:
        if inherited["predeclared"]!=expected:
            raise ValueError("A successor candidate must retain the predeclared holdout seed and indices")
        plan["inherited_holdout_declaration"]=str(out/"v2_evaluation_plan.json")
        plan["original_holdout_declared_at_kst"]=inherited["declared_at_kst"]
        plan["candidate_revision_reason"]="V2 failed an already observed seed221122 regression pilot; V3 retains the still-unused seed221123 holdout declaration."
    write_json(path,plan)
    return plan


def version_outputs(base,out,baseline,version,candidate_folder,final_folder,plan,historical_runs,generated):
    declared=plan["predeclared"]
    development=with_expectation(load_run(base/candidate_folder),declared["development_seed"],declared["development_indices"])
    held_out=with_expectation(load_run(base/final_folder),declared["held_out_seed"],declared["held_out_indices"])
    paired={"generated_at_kst":generated,**compare(baseline,development)}
    historical_hashes={row.get("scenario_sha256") for run in historical_runs for row in run["rows"].values()}
    new_hashes={row.get("scenario_sha256") for row in held_out["rows"].values()}
    final={"generated_at_kst":generated,**progress(held_out,development),
           "predeclared_plan":str(out/(version+"_evaluation_plan.json")),
           "previously_observed_scenario_hash_overlap":sorted((historical_hashes&new_hashes)-{None}),
           "evaluation_type":"Fresh-seed standalone holdout, not paired with the old-seed baseline."}
    final["valid_for_predeclared_frozen_holdout"]=(final["run_manifest_present"]
        and final["manifest_matches_declared_plan"] is True and final["evaluation_complete"]
        and final["frozen_code_across_completed_cases"] and final["frozen_model_across_completed_cases"]
        and not final["changed_during_case_indices"] and final["same_frozen_code_as_candidate"] is True
        and final["same_frozen_model_as_candidate"] is True
        and not final["previously_observed_scenario_hash_overlap"]
        and not final["development_scenario_hash_overlap"] and paired["valid_for_full_requested_comparison"])
    cover={"generated_at_kst":generated,"baseline_development":coverage(baseline),
           "candidate_development":coverage(development),"candidate_held_out":coverage(held_out)}
    summary={"version":version,"generated_at_kst":generated,"predeclared":declared,
             "development_comparison":paired,"held_out":final,
             "candidate_total_judged":paired["candidate"]["performance_attempts"]+final["performance_attempts"],
             "candidate_total_passed":paired["candidate"]["passed"]+final["passed"],
             "candidate_total_failed":paired["candidate"]["failed"]+final["failed"],
             "note":"Combined candidate counts are descriptive across development and holdout, not 20 paired trials or a held-out estimate."}
    write_json(out/("compare_"+version+".json"),paired)
    write_json(out/("final_"+version+"_progress.json"),final)
    write_json(out/("coverage_"+version+".json"),cover)
    return summary,development,held_out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--baseline",default="baseline")
    p.add_argument("--candidate",default="candidate-v1")
    p.add_argument("--final",dest="final_run",default="final-unseen")
    p.add_argument("--final-baseline",default="final-baseline")
    p.add_argument("--candidate-v2",default="candidate-v2")
    p.add_argument("--final-v2",default="final-v2")
    p.add_argument("--final-v2-seed",type=int,default=221123)
    p.add_argument("--final-v2-cases",type=int,default=12)
    p.add_argument("--candidate-v3",default="candidate-v3")
    p.add_argument("--final-v3",default="final-v3")
    p.add_argument("--v2-regression-pilot",default="v2-regression-case0")
    args=p.parse_args()
    base=args.base.resolve();out=base/"analysis";out.mkdir(parents=True,exist_ok=True)
    b=load_run(base/args.baseline);c=load_run(base/args.candidate);f=load_run(base/args.final_run)
    fb=load_run(base/args.final_baseline)
    generated=datetime.now(timezone(timedelta(hours=9))).isoformat()
    plan=version_plan(out,b["manifest"].get("seed",8122),b["requested"] or list(range(8)),
                      args.final_v2_seed,args.final_v2_cases,generated)
    paired={"generated_at_kst":generated,**compare(b,c)}
    final_paired={"generated_at_kst":generated,**compare(fb,f)}
    combined={"generated_at_kst":generated,**combine_suites([
        ("development",b,c,paired),("held_out",fb,f,final_paired)]),
        "plan_disposition":"Historical V1 plan stopped after regression evidence; planned_physical_runs records the original plan, not future pending work."}
    cover={"generated_at_kst":generated,"baseline":coverage(b),"candidate":coverage(c),
           "final_unseen":coverage(f),"final_baseline":coverage(fb)}
    final={"generated_at_kst":generated,**progress(f,c),
           "disposition":"V1 rejected after held-out regression; original 12-case suite stopped with history preserved."}
    pilot=load_run(base/args.v2_regression_pilot)
    pilot_summary={"generated_at_kst":generated,**run_summary(pilot),
        "disposition":"V2 regression pilot rejected; observed old-seed case, not a fresh holdout.",
        "records":[{"index":i,"suite_seed":row.get("suite_seed"),"success":row.get("success"),
                    "interrupted":bool(row.get("interrupted")),"failure_stage":row.get("failure_stage"),
                    "official_seconds":seconds(row),"scenario_sha256":row.get("scenario_sha256")}
                   for i,row in pilot["rows"].items()]}
    v2_summary,c2,f2=version_outputs(base,out,b,"v2",args.candidate_v2,args.final_v2,plan,(b,c,f,fb),generated)
    v2_summary["disposition"]="Rejected after old-seed regression pilot; not selected."
    v2_summary["observed_regression_pilot"]=pilot_summary
    plan_v3=version_plan(out,plan["predeclared"]["development_seed"],plan["predeclared"]["development_indices"],
                         args.final_v2_seed,args.final_v2_cases,generated,"v3",inherited=plan)
    v3_summary,c3,f3=version_outputs(base,out,b,"v3",args.candidate_v3,args.final_v3,plan_v3,(b,c,f,fb,c2,f2,pilot),generated)
    v3_dev=v3_summary["development_comparison"]
    v3_final=v3_summary["held_out"]
    if v3_final["valid_for_predeclared_frozen_holdout"]:
        v3_summary["disposition"]=("Verified frozen V3: development "
            +str(v3_dev["candidate"]["passed"])+"/"+str(v3_dev["candidate"]["performance_attempts"])
            +" and predeclared fresh holdout "+str(v3_final["passed"])+"/"+str(v3_final["performance_attempts"])
            +" passed; every declared case completed and artifact/plan checks passed.")
    elif v3_dev["comparison_complete"] and v3_final["evaluation_complete"]:
        v3_summary["disposition"]="V3 trials completed; frozen-artifact or declared-plan verification did not pass. Inspect validation fields."
    else:
        v3_summary["disposition"]="Current final candidate; outcome pending until all predeclared trials are reported."
    write_json(out/"compare.json",paired);write_json(out/"coverage.json",cover);write_json(out/"final_progress.json",final)
    write_json(out/"final_compare.json",final_paired);write_json(out/"combined_compare.json",combined)
    write_json(out/"v2_summary.json",v2_summary);write_json(out/"v2_regression_pilot.json",pilot_summary)
    write_json(out/"v3_summary.json",v3_summary)
    write_json(out/"latest_evaluation.json",v3_summary)
    print(json.dumps({"current_final_candidate":"v3","disposition":v3_summary["disposition"],
        "v3_development":{"pairs":v3_dev["pair_count"],"baseline_passed":v3_dev["baseline"]["passed"],
            "candidate_passed":v3_dev["candidate"]["passed"],"valid":v3_dev["valid_for_full_requested_comparison"],
            "gained":v3_dev["gained_pass_indices"],"lost":v3_dev["lost_pass_indices"],
            "shared_pass_seconds":v3_dev["official_seconds_shared_pass_only"]},
        "v3_fresh_holdout":{"seed":v3_final["suite_seed"],"requested":v3_final["requested"],
            "judged":v3_final["performance_attempts"],"passed":v3_final["passed"],
            "valid":v3_final["valid_for_predeclared_frozen_holdout"],
            "official_seconds":v3_final["official_seconds_success_only"]},
        "historical_v1":{"development_passed":paired["candidate"]["passed"],
            "holdout_passed":final["passed"],"holdout_judged":final["performance_attempts"],
            "holdout_lost_pass_indices":final_paired["lost_pass_indices"]},
        "historical_v2_pilot":{"passed":pilot_summary["passed"],"judged":pilot_summary["performance_attempts"]}},
        ensure_ascii=False))


if __name__=="__main__":main()
