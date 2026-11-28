"""Fast source-integrity and export gates; no training or deployed-file writes."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent

def verify_sources():
    expected=json.loads((root/"protected.json").read_text())
    for name,digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,("Protected source changed",name)
    return len(expected)

if __name__=="__main__":
    count=verify_sources()
    evidence=root/"final"
    train=json.loads((evidence/"training_result.json").read_text())
    export=json.loads((evidence/"export_result.json").read_text())
    contract=json.loads((root/"contract_result.json").read_text())
    rollouts=json.loads((evidence/"rollout_result.json").read_text())
    assert (evidence/"exit.txt").read_text().strip()=="0"
    assert (evidence/"evaluation_exit.txt").read_text().strip()=="0"
    assert train["changed_tensors"]>0 and export["max_abs_error"]<2e-5
    assert contract["max_abs_error"]<2e-6
    assert next(x for x in rollouts if x["policy"]=="supplied")["failure"] is None
    result={"protected_sources_unchanged":count,"training_and_export_pipeline_pass":True,
        "candidate_walking_pass":next(x for x in rollouts if x["policy"]=="candidate")["failure"] is None,
        "overnight_started":False,
        "known_transfer_differences":["MJWarp does not implement competition noslip_iterations=4",
            "Stationary idle root forces match official formulas; active walking has no root assistance",
            "nominal flat-floor stage; terrain, friction and sensor robustness still require evaluation"]}
    (root/"preflight_result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
