"""Read-only equivalence check for the registered pre-pilot engineering amendment."""
import argparse,csv,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('new',type=Path);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for root in (args.base,args.new):
    m=json.loads((root/'manifest.json').read_text());assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['profile']=='smoke'
    for name,binding in m['output_bindings'].items():assert (root/name).stat().st_size==binding['bytes'] and sha(root/name)==binding['sha256'],name
before=json.loads((args.base/'checkpoint_choices.json').read_text());after=json.loads((args.new/'checkpoint_choices.json').read_text());assert before.keys()==after.keys();count=0
for path in args.base.glob('N*/**/checkpoints/*.pt'):
    a=torch.load(path,weights_only=True);b=torch.load(args.new/path.relative_to(args.base),weights_only=True)
    assert a['parameter_sha256']==b['parameter_sha256'] and all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict']);count+=1
assert all(before[k]['steps']==after[k]['steps'] for k in before)
with (args.base/'episodes.csv').open() as f:a=list(csv.DictReader(f))
with (args.new/'episodes.csv').open() as f:b=list(csv.DictReader(f))
assert len(a)==len(b)
for left,right in zip(a,b):
    for k,v in left.items():
        if k!='decision_mean_ms':assert right[k]==v,k
out=args.output_root/('technician_scarcity_ppo_v1_engineering_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
report=dict(status='PASS',base=str(args.base),new=str(args.new),all_checkpoint_tensors_equal=count,all_original_episode_fields_equal=len(a),timing_excluded=True,script_sha256=sha(Path(__file__)))
(out/'execution_equivalence.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'verify_source.py');print(out)
