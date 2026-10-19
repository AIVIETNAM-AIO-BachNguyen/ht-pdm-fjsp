"""Independent artifact/statistics verification of the committed Mac full run."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import subprocess
import numpy as np

WORK=Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
sys.path.insert(0,str(WORK/'src'))
from ht_pdm_fjsp import maintenance_joint_frontier_mac as s
OUT=Path(__file__).resolve().parent
RUN=Path(Path('/private/tmp/marl_mac_frontier_full.txt').read_text().strip())
manifest=json.loads((RUN/'manifest.json').read_text());summary=json.loads((RUN/'summary.json').read_text());cfg=json.loads((RUN/'resolved_config.json').read_text())['settings']
assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS' and manifest['profile']=='full'
for name,binding in manifest['files'].items():
 p=RUN/name;assert p.stat().st_size==binding['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==binding['sha256'],name
for name in s.SOURCES:
 content=subprocess.check_output(['git','show',f"{manifest['git']['commit']}:src/ht_pdm_fjsp/{name}"],cwd=WORK)
 assert hashlib.sha256(content).hexdigest()==manifest['source_seal'][f'source_snapshot/{name}'],name
assert (RUN/'episodes.csv').read_bytes()==(RUN/'episodes.partial.csv').read_bytes()
with (RUN/'episodes.csv').open() as f: rows=list(csv.DictReader(f))
lookup={(r['method'],int(r['seed'])):r for r in rows}
assert len(rows)==1280 and len(lookup)==1280
assert set(lookup)=={(m,seed) for m in s.METHODS for seed in cfg['confirmation_seeds']}
assert not set(cfg['development_seeds'])&set(cfg['confirmation_seeds'])
selection=json.loads((RUN/'selection.json').read_text());assert hashlib.sha256((RUN/'selection.json').read_bytes()).hexdigest()==manifest['selection_sha256']
assert not selection['confirmation_outcomes_opened']
fields=('extended_cost','availability','terminal_pending','policy_cost')
values=np.array([[[float(lookup[m,seed][k]) for k in fields] for m in s.METHODS] for seed in cfg['confirmation_seeds']])
n=len(values);draws=np.random.default_rng(s.role_seed(cfg['analysis_seed'],f'{s.VERSION}:bootstrap')).integers(n,size=(cfg['bootstrap_samples'],n))
# Alternative bootstrap aggregation uses resample counts, not indexed value arrays.
weights=np.zeros((len(draws),n),np.int32)
for index,d in enumerate(draws): weights[index]=np.bincount(d,minlength=n)
boot=np.einsum('rs,smk->rmk',weights,values)/n;point=values.mean(0)
j=s.METHODS.index('joint_rollout');b=s.METHODS.index(selection['base']);q=[(1-cfg['confidence'])/2,(1+cfg['confidence'])/2]
checks={}
for m in s.CHEAP:
 i=s.METHODS.index(m);checks[f'cost_vs_{m}']=(1-point[j,0]/point[i,0],1-boot[:,j,0]/boot[:,i,0])
checks.update(availability_difference=(point[j,1]-point[b,1],boot[:,j,1]-boot[:,b,1]),pending_difference=(point[j,2]-point[b,2],boot[:,j,2]-boot[:,b,2]),absolute_availability=(point[j,1],boot[:,j,1]),absolute_pending=(point[j,2],boot[:,j,2]),interior_cost_gain=(1-point[j,3]/point[b,3],1-boot[:,j,3]/boot[:,b,3]))
for key,(estimate,samples) in checks.items():
 np.testing.assert_allclose(estimate,summary['comparisons'][key]['estimate'],atol=1e-12,rtol=0)
 np.testing.assert_allclose(np.quantile(samples,q),summary['comparisons'][key]['ci'],atol=1e-12,rtol=0)
assert len(checks)==8
for method in s.METHODS:
 expected={k:float(np.mean([float(r[k]) for r in rows if r['method']==method])) for k in rows[0] if k not in ('method','phase','seed','physical_seed')}
 actual=next(r for r in summary['means'] if r['method']==method)
 for k,v in expected.items():np.testing.assert_allclose(actual[k],v,atol=1e-10,rtol=0)
result=dict(status='PASS',verification='hash/commit/count/grid/selection seal and independent eight-CI bootstrap',run=str(RUN),files_checked=len(manifest['files']),commit=manifest['git']['commit'],scientific_status=summary['scientific_status'],comparisons=summary['comparisons'],gates=summary['gates'],means=summary['means'],mechanism=summary['mechanism'],subset_ablation=summary['subset_ablation'])
(OUT/'analysis.json').write_text(json.dumps(result,indent=2))
lines=['## Material Passport','','- Origin: ARS experiment-agent, independent statistics calculation in shared context.','- Verification: VERIFIED artifact/commit/CI calculations; physical replay comes from runner audit PASS.','','# Vòng 2: joint versus economic baseline frontier','',f"Verdict: **{summary['scientific_status']}**. Run `{RUN.name}`; commit `{manifest['git']['commit'][:7]}`. Fresh confirmation256 paired seeds, five methods, all1,280 episodes retained; development32 seeds frozen before test. Complete scalar replay of82,560 physical transitions; forecast replay limited to first2 confirmation seeds as registered. All file hashes and eight simultaneous99.375% intervals independently verified using count-weight bootstrap.",'','| Method | Cost H+L | H cost | Availability | Pending H |','|---|---:|---:|---:|---:|']
for r in summary['means']: lines.append(f"| {r['method']} | {r['extended_cost']:.3f} | {r['policy_cost']:.3f} | {r['availability']:.2%} | {r['terminal_pending']:.4f} |")
lines+=['','| Contrast | Estimate | CI99.375% | Gate |','|---|---:|---:|---|']
for key,r in summary['comparisons'].items():
 scale=100 if key not in ('pending_difference','absolute_pending') else 1;suffix='%' if scale==100 else ''
 lines.append(f"| {key} | {r['estimate']*scale:.4f}{suffix} | [{r['ci'][0]*scale:.4f}; {r['ci'][1]*scale:.4f}]{suffix} | {summary['gates'][key]} |")
lines+=['','Fixed physical profile only; known-model planning, restricted isolated DP is a capacity-relaxed surrogate. No inference of RL learnability, necessity of MARL, global optimality or assignment-isolated causality. Earlier assignment-primary failure stays unchanged. Per-protocol intervals are approximate percentile bootstrap; they do not give a global error guarantee for indefinite adaptive searches. All failed outcomes, controls and natural zero-disagreement landmarks retained.','', '11 fallacy checks: method-specific aggregation (Simpson/ecological); post-screen fixed profile disclosed (Berkson/regression to mean); all landmarks and ticks included (collider/base rate); complete episodes/losses kept (survivorship); eight locked simultaneous intervals (look elsewhere/forking paths); policy comparison separate from local mechanism/occupancy (causality/reverse causality). **11/11 checked.**']
(OUT/'report_vi.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'PASS','scientific_status':summary['scientific_status'],'gates':summary['gates'],'comparisons':summary['comparisons']},indent=2))
