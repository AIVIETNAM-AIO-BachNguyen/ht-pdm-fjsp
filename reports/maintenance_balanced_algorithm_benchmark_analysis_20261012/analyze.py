"""Read-only analysis of the completed, frozen lab benchmark; no retraining.

Primary intervals reproduce the registered calculation with explicit indices.
Mechanism/latency/replica tables and plots are descriptive, post-test analysis.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ALGORITHMS=('ppo','mappo','iql','qmix')
BASELINES=('deadline_matching','isolated_h6','isolated_endpoint','assignment_rollout','joint_rollout')
METRICS=('extended_cost','availability','terminal_pending')
VERSION='maintenance_balanced_algorithm_benchmark_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''): h.update(data)
    return h.hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def csvout(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read(path):
    with path.open() as stream: yield from csv.DictReader(stream)


def role(seed,kind):
    return int.from_bytes(hashlib.sha256(f'{seed}:{kind}'.encode()).digest()[:8],'big')


def run(source,out):
    source=source.resolve(); out=out.resolve()
    if out.exists(): raise FileExistsError('fresh analysis directory required')
    out.mkdir(parents=True)
    cfg=json.loads((source/'resolved_config.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    choices=json.loads((source/'checkpoint_choices.json').read_text())
    assert manifest['status']=='COMPLETED' and audit['status']=='PASS' and manifest['audit_status']=='PASS'
    assert manifest['git']['commit']=='2e190435bde46e99ac608ea19a3413a9bcc3a08c'
    for name,binding in manifest['output_bindings'].items():
        path=source/name
        assert path.stat().st_size==binding['bytes'] and sha(path)==binding['sha256'],name
    assert sha(source/'episodes.csv')==sha(source/'episodes.partial.csv')
    assert sha(source/'checkpoint_choices.json')==manifest['checkpoint_choices_sha256']
    assert (manifest['training_models'],manifest['training_learned_steps'],manifest['training_physical_ticks'],
        manifest['development_episodes'],manifest['test_episodes'])==(20,9830400,12288000,6400,6400)
    assert audit['physical_ticks']==384000 and audit['policy_records']==307200 and audit['invalid_assignments']==0
    assert len(list(source.glob('*/4600000*/checkpoints/*.pt')))==100

    rows=list(read(source/'episodes.csv')); lookup={(r['method'],int(r['train_seed']),int(r['seed'])):r for r in rows}
    expected={(a,t,s) for a in ALGORITHMS for t in cfg['train_seeds'] for s in cfg['test_seeds']}
    expected|={(b,-1,s) for b in BASELINES for s in cfg['test_seeds']}
    assert len(rows)==len(lookup)==6400 and set(lookup)==expected
    populations={}
    for r in rows:
        seed=int(r['seed']); physical=role(seed,f'{VERSION}:physical')
        assert int(r['physical_seed'])==physical and int(r['raw_seed'])==seed and not r['stream_seed']
        assert populations.setdefault(seed,r['population_sha256'])==r['population_sha256']
        assert float(r['invalid_assignments'])==0 and int(r['physical_ticks'])==60
        assert abs(float(r['policy_cost'])+float(r['tail_cost'])-float(r['extended_cost']))<1e-8
        assert abs(sum(float(r[k]) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost'))-float(r['extended_cost']))<1e-8

    dev=list(read(source/'development_episodes.csv'))
    devlookup=defaultdict(list)
    expecteddev={(a,t,k,s) for a in ALGORITHMS for t in cfg['train_seeds'] for k in (0,*cfg['checkpoints']) for s in cfg['development_seeds']}
    assert len(dev)==6400 and {(r['method'],int(r['train_seed']),int(r['checkpoint_steps']),int(r['seed'])) for r in dev}==expecteddev
    for r in dev: devlookup[r['method'],int(r['train_seed']),int(r['checkpoint_steps'])].append(float(r['extended_cost']))
    assert len(choices)==20
    for a,t in ((a,t) for a in ALGORITHMS for t in cfg['train_seeds']):
        means={k:float(np.mean(devlookup[a,t,k])) for k in cfg['checkpoints']}
        k=min(means,key=lambda k:(means[k],k)); chosen=choices[f'{a}:{t}']
        assert k==chosen['steps'] and abs(means[k]-chosen['development_cost'])<1e-10
        assert sha(source/chosen['path'])==chosen['sha256']

    train_counts=Counter(); train_population={}; train_identities=set()
    for r in tqdm(read(source/'training_episodes.csv'),total=204800,desc='Check paired training populations',unit='episode'):
        a=r['algorithm']; t=int(r['train_seed']); w=int(r['worker']); e=int(r['episode_index'])
        identity=a,t,w,e; assert identity not in train_identities; train_identities.add(identity)
        assert a in ALGORITHMS and t in cfg['train_seeds'] and 0<=w<16 and 0<=e<640
        phys=role(t,f'{VERSION}:training_physical:worker:{w}:episode:{e}')
        assert int(r['physical_seed'])==phys and int(r['stream_seed'])==t and not r['raw_seed']
        assert train_population.setdefault((t,w,e),(phys,r['population_sha256']))==(phys,r['population_sha256'])
        assert int(r['physical_ticks'])==60 and int(r['learned_episode_steps'])==48
        assert abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5
        train_counts[a,t]+=1
    assert len(train_identities)==204800 and len(train_population)==51200 and all(v==10240 for v in train_counts.values())
    training=json.loads((source/'training_audit.json').read_text())
    progress={}
    for r in read(source/'training_progress.csv'):
        a=r['algorithm']; t=int(r['train_seed']); steps=int(r['learned_steps']); progress[a,t]=r
        if steps in cfg['checkpoints'] and choices[f'{a}:{t}']['steps']==steps:
            assert r['parameter_sha256']==choices[f'{a}:{t}']['parameter_sha256']
    for r in training:
        a=r['algorithm']; t=r['train_seed']; last=progress[a,t]
        assert r['optimizer_steps']==(7680 if a in ('ppo','mappo') else 30529)
        assert r['changed'] and r['partial_episode_steps']==0 and r['learned_steps']==491520
        assert int(last['learned_steps'])==491520 and last['parameter_sha256']==r['parameter_sha256']

    learned=np.asarray([[[[float(lookup[a,t,s][m]) for m in METRICS] for s in cfg['test_seeds']]
        for t in cfg['train_seeds']] for a in ALGORITHMS])
    baseline=np.asarray([[[float(lookup[b,-1,s][m]) for m in METRICS] for s in cfg['test_seeds']] for b in BASELINES])
    points={}; draws={}
    lm=learned.mean((1,2)); bm=baseline.mean(1)
    for i,a in enumerate(ALGORITHMS):
        for j,b in enumerate(BASELINES): points[f'{a}_cost_vs_{b}']=1-lm[i,0]/bm[j,0]
        points[f'{a}_availability_difference']=lm[i,1]-bm[1,1]
        points[f'{a}_pending_difference']=lm[i,2]-bm[1,2]
    draws={key:np.empty(cfg['bootstrap_samples']) for key in points}
    bootstrap_seed=role(cfg['analysis_seed'],f'{VERSION}:crossed_bootstrap')
    rng=np.random.default_rng(bootstrap_seed)
    # Same registered RNG consumption; different explicit Cartesian index estimator.
    for start in tqdm(range(0,cfg['bootstrap_samples'],256),desc='Independently verify 28 bootstrap intervals',unit='batch'):
        count=min(256,cfg['bootstrap_samples']-start)
        ti=rng.integers(5,size=(count,5)); si=rng.integers(256,size=(count,256))
        for k in range(count):
            lm=learned[:,ti[k]][:,:,si[k]].mean((1,2)); bm=baseline[:,si[k]].mean(1)
            for i,a in enumerate(ALGORITHMS):
                for j,b in enumerate(BASELINES): draws[f'{a}_cost_vs_{b}'][start+k]=1-lm[i,0]/bm[j,0]
                draws[f'{a}_availability_difference'][start+k]=lm[i,1]-bm[1,1]
                draws[f'{a}_pending_difference'][start+k]=lm[i,2]-bm[1,2]
    q=(1-cfg['confidence'])/2; max_error=0.; verified_intervals={}
    for key,point in points.items():
        lo,hi=np.quantile(draws[key],[q,1-q]); registered=summary['intervals'][key]
        for field,value in [('estimate',point),('lower',lo),('upper',hi)]:
            error=abs(value-registered[field]); max_error=max(max_error,error); assert error<1e-10,(key,field,error)
        verified_intervals[key]=dict(estimate=point,lower=float(lo),upper=float(hi))
    assert len(verified_intervals)==28

    groups=defaultdict(list)
    for r in rows: groups[r['method']].append(r)
    metrics=['extended_cost','availability','terminal_pending','tail_pending','policy_cost','tail_cost',
        'maintenance_cost','unavailability_cost','failure_cost','waiting_cost','policy_failures',
        'conflict_workers','rejected_proposals','unused_feasible_capacity','decision_seconds']
    table=[]; replicas=[]
    for method in (*ALGORITHMS,*BASELINES):
        means={m:float(np.mean([float(r[m]) for r in groups[method]])) for m in metrics}
        table.append(dict(method=method,**means,cost_gain_vs_h6=1-means['extended_cost']/summary['means']['isolated_h6']['extended_cost'],
            decision_ms=1000*means['decision_seconds']/48))
        for t in (cfg['train_seeds'] if method in ALGORITHMS else [-1]):
            costs=[float(lookup[method,t,s]['extended_cost']) for s in cfg['test_seeds']]
            replicas.append(dict(method=method,train_seed=t,test_mean_cost=float(np.mean(costs)),
                gain_vs_h6=1-float(np.mean(costs))/summary['means']['isolated_h6']['extended_cost'],
                selected_steps=choices[f'{method}:{t}']['steps'] if method in ALGORITHMS else -1))
    mechanism={m:Counter() for m in (*ALGORITHMS,*BASELINES)}
    matching_hash={}; matching_h=hashlib.sha256(); trajectory_key=None
    for r in tqdm(read(source/'testdecisions.csv'),total=384000,desc='Describe technician assignment',unit='tick'):
        method=r['method']; tick=int(r['tick']); key=method,int(r['train_seed']),int(r['seed'])
        if tick==0: matching_h=hashlib.sha256(); trajectory_key=key
        assert key==trajectory_key
        if tick<48:
            counter=mechanism[method]; state=json.loads(r['state']); action=json.loads(r['action'])
            matching_h.update(json.dumps(action,separators=(',',':')).encode())
            counter['ticks']+=1; counter['empty_matching_ticks']+=int(not action)
            for machine,worker in action:
                fast=(machine<2 and worker==0) or (machine>=2 and worker==1)
                counter['starts']+=1; counter['fast_starts']+=fast; counter['slow_starts']+=not fast
                counter[f'm{machine}_t{worker}']+=1
                counter['corrective_starts']+=state['failed'][machine]
                counter['preventive_starts']+=not state['failed'][machine]
        if tick==59: matching_hash[key]=matching_h.hexdigest()
    trajectories=[]
    for a in ALGORITHMS:
        per_replica={t:tuple(matching_hash[a,t,s] for s in cfg['test_seeds']) for t in cfg['train_seeds']}
        unique=len(set(per_replica.values()))
        trajectories.append(dict(method=a,unique_behavior_panels_across_five_replicas=unique,
            note='identical observed trajectories do not imply identical weights'))
    mechanisms=[]
    for m,c in mechanism.items():
        n=len(groups[m]); row=dict(method=m,starts_per_episode=c['starts']/n,
            fast_share=c['fast_starts']/c['starts'],slow_share=c['slow_starts']/c['starts'],
            slow_starts_per_episode=c['slow_starts']/n,preventive_starts_per_episode=c['preventive_starts']/n,
            corrective_starts_per_episode=c['corrective_starts']/n,empty_matching_fraction=c['empty_matching_ticks']/c['ticks'])
        row.update({f'm{machine}_t{worker}_starts_per_episode':c[f'm{machine}_t{worker}']/n for machine in range(4) for worker in range(2)})
        mechanisms.append(row)
    verification=dict(status='ANALYZED',artifact_integrity='PASS',exact_grids='PASS',checkpoint_selection='PASS',
        training_population_pairing='PASS',registered_statistics_recalculation='PASS',registered_ci_count=28,
        bootstrap_draws=40000,maximum_ci_difference=max_error,bound_files=len(manifest['output_bindings']),
        bound_bytes=sum(v['bytes'] for v in manifest['output_bindings'].values()),
        source_manifest_sha256=sha(source/'manifest.json'),training_rerun=False,
        physical_replay='original lab audit PASS; no new full simulator replay on Mac',
        source=str(source),analysis_script_sha256=sha(Path(__file__)),elapsed_lab_seconds=manifest['elapsed_seconds'])
    write(out/'verification.json',verification); write(out/'registered_intervals_verified.json',verified_intervals)
    write(out/'analysis.json',dict(verification=verification,table=table,replicas=replicas,
        mechanisms=mechanisms,trajectory_diversity=trajectories,guards=summary['guards'],
        development_curves={f'{a}:{t}:{k}':float(np.mean(v)) for (a,t,k),v in devlookup.items()},
        training=training,all_mechanism_findings='descriptive/exploratory; no extra confirmatory intervals'))
    csvout(out/'method_summary.csv',table); csvout(out/'replica_summary.csv',replicas); csvout(out/'technician_assignments.csv',mechanisms)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'ppo':'#2468b4','mappo':'#e18a28','iql':'#b84851','qmix':'#418965'}
    fig,ax=plt.subplots(figsize=(8,3.6))
    for i,a in enumerate(ALGORITHMS):
        v=verified_intervals[f'{a}_cost_vs_isolated_h6']
        ax.errorbar(100*v['estimate'],i,xerr=[[100*(v['estimate']-v['lower'])],[100*(v['upper']-v['estimate'])]],
            fmt='o',capsize=5,color=colors[a],markersize=7)
    ax.axvline(0,color='#555',lw=1); ax.axvline(1,color='#999',lw=1,ls=':')
    ax.set_yticks(range(4),[a.upper() for a in ALGORITHMS]); ax.invert_yaxis()
    ax.set_xlabel('Cost reduction vs isolated h6 (%) — positive is better')
    ax.set_title('Registered crossed-bootstrap intervals (99.8214% each)')
    ax.grid(axis='x',alpha=.15); fig.tight_layout(); fig.savefig(out/'cost_vs_h6.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    steps=[0,*cfg['checkpoints']]
    for ax,a in zip(axes.ravel(),ALGORITHMS):
        curves=np.asarray([[np.mean(devlookup[a,t,k]) for k in steps] for t in cfg['train_seeds']])
        for curve in curves: ax.plot(np.asarray(steps)/1000,curve,color=colors[a],lw=1,alpha=.3)
        ax.plot(np.asarray(steps)/1000,curves.mean(0),color=colors[a],lw=2.5,marker='o')
        ax.set_title(a.upper()); ax.set_ylabel('Development H+L cost'); ax.set_xlabel('Learned transitions (thousands)'); ax.grid(alpha=.15)
    fig.suptitle('Development only: thin lines = five training replicas; thick line = mean',fontsize=11)
    fig.tight_layout(); fig.savefig(out/'development_learning_curves.png',dpi=180); plt.close(fig)
    print(json.dumps(dict(verification=verification,methods=table,mechanisms=mechanisms,trajectory_diversity=trajectories),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.source,args.output)
