"""Exercise stock --llm-web through its request-file protocol, without UI clicks."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import yaml


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'custom/docs/validation'
    out.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((root/'main_config.yaml').read_text())
    config['simulator']['view']=False
    config['instruction']['open_browser']=False
    config['instruction']['web_port']=18501
    instruction='Walk to (15.5, 18.0), then go to the green cone.'
    state=root/'web/state'
    request_id=str(uuid.uuid4())
    progress={}
    with tempfile.TemporaryDirectory(prefix='rai-web-eval-') as directory:
        cfg=Path(directory)/'config.yaml'
        cfg.write_text(yaml.safe_dump(config))
        with (out/'web_smoke.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(root/'main.py'),'--llm-web','--config',str(cfg)],
                cwd=root,env={**os.environ,'MUJOCO_GL':'egl','HAIC_REQUIRE_LLM':'1','PYTHONDONTWRITEBYTECODE':'1'},
                stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+35
                while time.monotonic()<deadline and proc.poll() is None:
                    if 'type the instruction on' in (out/'web_smoke.log').read_text(): break
                    time.sleep(.3)
                else: raise RuntimeError('stock web mode did not become ready')
                state.mkdir(parents=True,exist_ok=True)
                temp=state/'request.smoke.tmp'
                temp.write_text(json.dumps({'id':request_id,'instruction':instruction}))
                os.replace(temp,state/'request.json')
                deadline=time.monotonic()+210
                while time.monotonic()<deadline and proc.poll() is None:
                    try: progress=json.loads((state/'progress.json').read_text())
                    except (OSError,ValueError): progress={}
                    if progress.get('id')==request_id and progress.get('state') in ('done','error'): break
                    time.sleep(.4)
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
    (out/'web_smoke.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps(progress,indent=2))
    if progress.get('state')!='done' or progress.get('total',{}).get('passed')!=2:
        raise SystemExit(1)


if __name__=='__main__':
    main()
