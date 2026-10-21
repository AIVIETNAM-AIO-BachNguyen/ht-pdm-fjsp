"""Pre-full V19 feature-discrimination check on already-opened V18 DEV states."""
from __future__ import annotations

import argparse,csv,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,configs
from ht_pdm_fjsp.maintenance_framework_v18_campaign import legal_present
from ht_pdm_fjsp.maintenance_framework_v19_policy import matching_profiles
from ht_pdm_fjsp.maintenance_requests import state_from_dict

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research_result/framework_v18_dev_20261021T050413Z/counterfactual_states.csv'
PLAN=ROOT/'research_result/framework_v19_dev_protocol_vi.md'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cells={}
    with SOURCE.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['arm']!='bounded_disagreement' or int(row['iteration'])!=1:continue
            bin_=min(int(row['tick'])//16,2)
            cells.setdefault((row['case'],bin_),[]).append(row)
    rows=[]
    for case in CASES:
        for bin_,take in enumerate((11,11,10)):
            selected=sorted(cells[(case,bin_)],key=lambda x:x['rank'])[:take]
            assert len(selected)==take
            for row in selected:
                tick=int(row['tick']);state=state_from_dict(json.loads(row['original_state']))
                _,_,legal,obs=legal_present(configs()[case],state,tick)
                assert legal==tuple(json.loads(row['legal']))
                temporal=matching_profiles(obs.reshape(1,56),'temporal_deficit')[0]
                snapshot=matching_profiles(obs.reshape(1,56),'snapshot_deficit')[0]
                delta=temporal[list(legal),9:]-snapshot[list(legal),9:]
                any_change=bool(np.any(delta!=0))
                action_varying=bool(len(legal)>1 and np.any(delta!=delta[:1]))
                rows.append(dict(case=case,clock_bin=bin_,train_seed=row['train_seed'],tick=tick,
                    rank=row['rank'],physical_seed=row['physical_seed'],legal_actions=len(legal),
                    any_change=int(any_change),action_varying=int(action_varying),
                    changed_action_count=int(np.count_nonzero(np.any(delta!=0,axis=1)))))
    with (out/'states.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='COMPLETED',sampled_states=len(rows),changed_states=sum(r['any_change'] for r in rows),
        action_varying_states=sum(r['action_varying'] for r in rows),
        by_case={case:dict(sampled=sum(r['case']==case for r in rows),changed=sum(r['any_change'] for r in rows if r['case']==case),
            action_varying=sum(r['action_varying'] for r in rows if r['case']==case)) for case in CASES})
    summary['sanity_gate']=summary['changed_states']/len(rows)>.01
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest=dict(status='COMPLETED',utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sha(SOURCE),plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        policy_sha256=sha(ROOT/'src/ht_pdm_fjsp/maintenance_framework_v19_policy.py'),
        artifact_hashes={name:sha(out/name) for name in ('states.csv','summary.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.output_dir)))
if __name__=='__main__':main()
