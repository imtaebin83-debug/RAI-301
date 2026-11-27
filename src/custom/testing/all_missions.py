"""Integration-only evaluator. Without --run it prints the frozen case manifest."""
import argparse
import json
from pathlib import Path
from custom.mission_2.evaluate import run_case


def cases():
    values = [{'id':f'mission-{n}', 'missions':[n], 'split':'integration-development'}
              for n in range(1,7)]
    values.append({'id':'all-six-blue', 'missions':[1,2,3,4,5,6],
                   'split':'integration-development', 'm1_goal':[15.5,18.,90.],
                   'm2_color':'blue', 'cone_order':['orange','green','blue'],
                   'room_order':['medical','cafeteria','classroom']})
    values.append({**values[-1], 'id':'finish-4-5-6', 'missions':[4,5,6]})
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    selected=[case for case in cases() if case['id'] in args.case]
    if set(args.case)-{case['id'] for case in selected}:
        parser.error('unknown integration case')
    if not args.run:
        print(json.dumps(selected,indent=2))
        return
    if args.output is None:
        parser.error('--output is required for actual execution')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    records=[]
    for case in selected:
        record=run_case(case,root,out,2400. if len(case['missions'])>1 else 900.)
        records.append(record)
        (out/'summary.json').write_text(json.dumps(records,indent=2))
        if record['returncode']==75 or 'interrupted' in record:
            raise SystemExit(record['returncode'] or 1)
    if any(not row.get('official_result',{}).get('total',{}).get('complete')
           or row['official_result']['total'].get('failed',1) for row in records):
        raise SystemExit(1)


if __name__=='__main__':
    main()
