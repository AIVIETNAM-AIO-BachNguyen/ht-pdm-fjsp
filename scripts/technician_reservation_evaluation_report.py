"""Read-only evidence report; sealed completed inputs only."""
import argparse,csv,hashlib,json,math,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--torch-python',type=Path);args=p.parse_args()
source=args.source.resolve();manifest=json.loads((source/'manifest.json').read_text())
if manifest['status']!='COMPLETED' or manifest.get('audit_status')!='PASS':raise SystemExit('Requires completed audited inputs')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
inputs=['manifest.json','summary.json','resolved_config.json','audit.json','episodes.csv','checkpoint_choices.json','warm_choices.json','teachers/references.json']
if (source/'runtime_environment.json').exists():inputs.append('runtime_environment.json')
for name in inputs:
 if name!='manifest.json':
  binding=manifest['output_bindings'][name]
  if sha(source/name)!=binding['sha256']:raise SystemExit('Sealed input changed: '+name)
s=json.loads((source/'summary.json').read_text());cfg=json.loads((source/'resolved_config.json').read_text());audit=json.loads((source/'audit.json').read_text());choices=json.loads((source/'checkpoint_choices.json').read_text())
with (source/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
if len(rows)!=manifest['test_episodes'] or any(float(r['invalid_assignments']) != 0 for r in rows):raise SystemExit('Episode audit disagreement')
scientific=s['classification']=='CONFIRMATORY' and cfg['profile']=='confirmation'
if manifest['protocol']!='technician_reservation_ppo_v4':raise SystemExit('This report covers the verified V4 architecture only')
parameter_paths=[(n,arm,str(source/choices[f'{n}:{arm}:{t}']['path'])) for n in cfg['scales'] for arm in ('reservation_ppo','independent_ppo') for t in cfg['train_seeds']]
for _,_,name in parameter_paths:
 relative=str(Path(name).relative_to(source))
 if sha(name)!=manifest['output_bindings'][relative]['sha256']:raise SystemExit('Selected checkpoint changed')
parameter_probe="import torch,json,sys; torch.set_num_threads(1); paths=json.load(sys.stdin); print(json.dumps([sum(t.numel() for t in torch.load(p,map_location='cpu',weights_only=True)['state_dict'].values()) for p in paths]))"
torch_python=args.torch_python or source.parent.parent/'.venv/bin/python'
counts=json.loads(subprocess.run([str(torch_python),'-c',parameter_probe],input=json.dumps([p for _,_,p in parameter_paths]),text=True,capture_output=True,check=True).stdout)
parameter_counts={n:{count for (size,_,_),count in zip(parameter_paths,counts) if size==n} for n in cfg['scales']}
if any(len(c)!=1 for c in parameter_counts.values()) or len(set(counts))!=1:raise SystemExit('Replica/scale parameter counts differ')
parameter_counts={n:next(iter(c)) for n,c in parameter_counts.items()}
inputs.extend(str(Path(path).relative_to(source)) for _,_,path in parameter_paths)
kind='SCIENTIFIC_EVIDENCE' if scientific else 'ENGINEERING_RENDER_QA'
out=args.output_root.resolve()/f'{manifest["protocol"]}_evidence_report_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}';out.mkdir(parents=True)
scales=cfg['scales'];labels={'reservation_ppo':'Reservation PPO','independent_ppo':'Independent PPO','ppo':'Central PPO','mappo':'MAPPO','deadline':'Deadline'}
text=['# Technician reservation PPO evaluation','','Classification: **'+kind+'**. Source: `'+str(source)+'`.','Source Git: `'+manifest['git']['commit']+'`. Audit: PASS.',f'Required predeclared gates pass: **{s["goal_pass"]}**.']
if not scientific:text+=['','This engineering smoke report validates rendering and provenance; it supports no scientific conclusion.']
text+=['','## Cost and system health','','Cost includes 48 policy decisions plus 12 common deadline-tail ticks. Availability,','pending and failure guards use the policy cutoff. All training bundles are retained.','','| N/K | Method | Cost/machine | Availability | Pending/machine | Failures/machine |','|---|---|---:|---:|---:|---:|']
for n in scales:
 for arm,m in s['means'][str(n)].items():
  text.append(f'| {n}/{n//2} | {labels[arm]} | {m["normalized_cost"]:.4f} | {m["availability"]:.6f} | {m["normalized_pending"]:.6f} | {m["normalized_failures"]:.6f} |')
text+=['','## Predeclared intervals and gates','',f'Crossed bootstrap: {cfg["bootstrap_samples"]} draws, {len(cfg["train_seeds"])} bundled teacher/student replicas and {len(cfg["test_seeds"])} paired episode seeds per scale.','Episode axes are resampled independently by scale; replica weights are shared','across scales. This is conditional on the existing five-teacher collection.',f'Family: {s["family_size"]} endpoints. Scientific round {cfg["confirmation_round"]}, alpha {cfg["round_alpha"]}; individual two-sided confidence {s["confidence"]:.9f}.','Cost gains and availability differences are fractions; backlog/failure differences use per-machine units. Ratios are dimensionless.','Bonferroni intervals are approximate with five bundles; replica/leave-one-out','sensitivity is reported separately. Engineering smoke spends no scientific alpha.','','| Endpoint | Estimate | Lower | Upper | Locked condition | Pass |','|---|---:|---:|---:|---|---|']
for key,i in s['intervals'].items():
 if key.endswith('cost_ratio_vs_N4'):threshold=1.10;direction='upper';strict=False
 elif 'availability_difference' in key:threshold=-.01;direction='lower';strict=False
 elif 'pending_per_machine_difference' in key:threshold=.025;direction='upper';strict=False
 elif 'failures_per_machine_difference' in key:threshold=.25;direction='upper';strict=False
 elif key=='N4_gain_vs_ppo' or key=='N4_gain_vs_independent_ppo':threshold=-.01;direction='lower';strict=False
 elif key.endswith('gain_vs_deadline'):threshold=-.02;direction='lower';strict=False
 else:threshold=0.;direction='lower';strict=True
 value=i[direction];passed=value<threshold if direction=='upper' and strict else value<=threshold if direction=='upper' else value>threshold if strict else value>=threshold
 symbol='<' if direction=='upper' and strict else '<=' if direction=='upper' else '>' if strict else '>='
 text.append(f'| {key} | {i["estimate"]:.7f} | {i["lower"]:.7f} | {i["upper"]:.7f} | {direction} {symbol} {threshold} | {passed} |')
text+=['',f'Optional practical PPO superiority (cost gain lower bound >= 1%): **{s["guards"]["practical_ppo_superiority"]}**.',f'Optional positive small-setting coordination effect: **{s["guards"]["small_mechanism"]}**.','The required N4 ablation condition is noninferiority within 1%; positive','coordination effects are required separately at every larger setting.','','## Inference and structural scaling','','| N/K | Parameters | Neural p95 (ms) | Neural p99 (ms) | Samples | Full matching catalogue (theoretical) |','|---|---:|---:|---:|---:|---:|']
for n in scales:
 lat=s['latency'][f'{n}:reservation_ppo'];count=sum(math.comb(n,r)*math.comb(n//2,r)*math.factorial(r) for r in range(n//2+1))
 text.append(f'| {n}/{n//2} | {parameter_counts[n]} | {lat["p95_ms"]:.3f} | {lat["p99_ms"]:.3f} | {lat["samples"]} | {count} |')
text+=['','The locked neural p95 limit is 50 ms; zero invalid assignments is independently','audited. Latency measures `model.select` on this lab after training stops.','Observation construction, physical stepping/arbitration and network transport','are outside neural timing. Catalogue counts are structural, not runtime','speedup measurements. The graph decoder/tail at N>4 use no matching catalogue;','the teacher bank and centralized reference use only the small N4 catalogue.','','## Mechanism diagnostics','','Matched graph arms share complete raw graph information, public IDs/order,','capacity, initialization, teacher bank/augmentation and training budgets. The','intervention removes reservation/prefix exchange. The contrast concerns that','whole service; it does not isolate attention, augmentation or value calibration.','Descriptive diagnostics below add no hypothesis tests.','','| N | Arm | Conflicts/episode | Rejected proposals/episode | Pending/machine | Failures/machine | Availability |','|---|---|---:|---:|---:|---:|---:|']
if 'runtime_environment.json' in inputs:
 hardware=json.loads((source/'runtime_environment.json').read_text())
 text.insert(text.index('## Mechanism diagnostics'),f'Hardware: {hardware["cpu_model"]}, {hardware["logical_cpus"]} logical CPUs, {hardware["mem_total_gib"]:.2f} GiB RAM; CPU execution, one torch thread per worker. Packages: `{hardware["packages"]}`.\n')
colors={'reservation_ppo':'#1764ab','independent_ppo':'#d46a1c'};metrics=[('conflicts','Conflicts / episode',1),('normalized_pending','Pending / machine',1),('normalized_failures','Failures / machine',1),('availability','Availability (%)',100)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,4,figsize=(14,4),squeeze=False)
for n in scales:
 for arm in colors:
  panel=[r for r in rows if int(r['n'])==n and r['method']==arm]
  means={k:np.mean([float(r[k]) for r in panel]) for k in ('conflicts','rejected_proposals','normalized_pending','normalized_failures','availability')}
  text.append(f'| {n} | {labels[arm]} | {means["conflicts"]:.3f} | {means["rejected_proposals"]:.3f} | {means["normalized_pending"]:.6f} | {means["normalized_failures"]:.6f} | {means["availability"]:.6f} |')
for ax,(metric,label,multiplier) in zip(axs[0],metrics):
 for ai,arm in enumerate(colors):
  for ni,n in enumerate(scales):
   panel=[r for r in rows if int(r['n'])==n and r['method']==arm];seeds=sorted({r['train_seed'] for r in panel});replicas=np.array([np.mean([float(r[metric])*multiplier for r in panel if r['train_seed']==seed]) for seed in seeds]);x=ni+(ai-.5)*.34
   ax.bar(x,replicas.mean(),width=.30,color=colors[arm],alpha=.65,label=labels[arm] if ni==0 else None)
   ax.scatter(x+np.linspace(-.08,.08,len(replicas)),replicas,s=20,color=colors[arm],edgecolors='white',linewidths=.5,zorder=3)
 ax.set_xticks(range(len(scales)),[str(n) for n in scales]);ax.set_xlabel('Machines');ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
axs[0,0].legend(fontsize=8);fig.suptitle(('Sealed scientific tests' if scientific else 'Engineering-only rendering QA')+' — descriptive mechanism diagnostics',fontsize=12);fig.text(.5,.005,'Bars: all-bundle means. Dots: every replica. No additional hypothesis tests or confidence intervals in this figure.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(out/'mechanism_system_health.png',dpi=180);fig.savefig(out/'mechanism_system_health.pdf');plt.close(fig)
text+=['','## Replica and checkpoint sensitivity','','| N | Arm | Student seed | Selected step | Cost/machine | Availability | Pending/machine | Failures/machine |','|---|---|---:|---:|---:|---:|---:|---:|']
for n in scales:
 for arm in colors:
  for t in cfg['train_seeds']:
   panel=[r for r in rows if int(r['n'])==n and r['method']==arm and int(r['train_seed'])==t];means={k:np.mean([float(r[k]) for r in panel]) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')};step=choices[f'{n}:{arm}:{t}']['steps']
   text.append(f'| {n} | {labels[arm]} | {t} | {step} | {means["normalized_cost"]:.4f} | {means["availability"]:.6f} | {means["normalized_pending"]:.6f} | {means["normalized_failures"]:.6f} |')
loo=s['leave_one_replica_out'];keys=['N4_gain_vs_ppo',*[f'N{n}_gain_vs_independent_ppo' for n in scales if n!=4]]
text+=['','Leave-one-replica-out point-estimate ranges, without reselection or new tests:','','| Endpoint | Minimum | Maximum |','|---|---:|---:|']
for key in keys:
 vals=[x['estimates'][key] for x in loo]
 if vals:text.append(f'| {key} | {min(vals):.7f} | {max(vals):.7f} |')
text+=['','## Budgets, provenance and scope','',f'Completed model jobs: {audit["training_models"]}, including {audit["reference_models"]} frozen references. Training episodes: {audit["training_episodes"]}; paired training tapes: {audit["paired_training_tapes"]}.',f'Teacher state exposures: {audit["teacher_bank_exposures"]}. Every graph arm/student bundle uses {cfg["teacher_bank_steps"]} shared teacher states, {cfg["teacher_bank_steps"]*cfg["teacher_epochs"]} pretraining presentations, and {cfg["env_steps"]} fresh PPO decisions per scale.',f'The source foundation PPO/MAPPO each spent 491520 original decisions; reference checkpoints were selected on their original development panel and never reselected on these tests. {cfg["training_processes"]} independent single-thread CPU workers were used; execution parallelism is not an algorithm training-speed claim.',f'All {len(rows)} final episodes and {audit["test_physical_ticks_replayed"]} physical ticks were independently audited. Choices were sealed at `{manifest.get("choices_sealed_utc")}` before test generation.','Teacher assistance, larger graph capacity and richer information than local MAPPO','are disclosed. Results support no optimizer-only or sample-efficiency claim over','PPO/MAPPO. The matched graph ablation isolates the coordination service.','The scope is the balanced maintenance benchmark, fixed 48+12 horizon and','registered N/K ratios (technicians=N/2), with shared graph/reservation service.','Field deployment, network latency and arbitrary FJSP workload/skill ratios are','outside this evaluation.','',f'Generated{datetime.now(timezone.utc).isoformat()}; read-only report with sealed input hashes.']
(out/'evaluation_report.md').write_text('\n'.join(text)+'\n');shutil.copyfile(__file__,out/'report_source.py')
metadata=dict(classification=kind,source=str(source),input_hashes={name:sha(source/name) for name in inputs},report_source_sha256=sha(__file__),matplotlib=matplotlib.__version__,numpy=np.__version__,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()});(out/'report_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(out)
