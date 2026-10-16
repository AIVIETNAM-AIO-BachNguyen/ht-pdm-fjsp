"""Posthoc independent arithmetic and27 direct actor-parameter score checks."""
import argparse,csv,hashlib,importlib.util,itertools,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
TRAINS=tuple(range(280000000,280000003));DEV=tuple(range(280100000,280100032));STEPS=tuple(range(0,983041,122880))
def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def near(a,b,label):check(np.allclose(a,b,atol=2e-10,rtol=2e-10),label)
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))


def direct_fisher(inputs,weights,legal,ranks):
    z=torch.as_tensor(inputs,dtype=torch.float64)
    for layer in (0,2,4):
        z=torch.nn.functional.linear(z,weights[f'actor.{layer}.weight'],weights[f'actor.{layer}.bias'])
        if layer<4:z=z.tanh()
    k,width=z.shape;n=width-1;mask=torch.as_tensor(np.c_[np.ones(k,bool),legal.T]);logp=torch.log_softmax(z.masked_fill(~mask,-torch.inf),1)
    groups={};proposal_trace=0.
    supports=[[0]+[m+1 for m in range(n) if legal[m,j]] for j in range(k)]
    for u in itertools.product(*supports):
        logmass=sum(logp[j,t] for j,t in enumerate(u));p=float(logmass.detach().exp());grad=torch.autograd.grad(logmass,tuple(weights.values()),retain_graph=True)
        score=torch.cat([g.flatten() for g in grad]).detach().numpy();out=[0]*k
        for token in set(u)-{0}:
            winner=min((j for j,t in enumerate(u) if t==token),key=lambda j:ranks[token-1,j]);out[winner]=token
        key=tuple(out)
        if key not in groups:groups[key]=[0.,np.zeros_like(score)]
        groups[key][0]+=p;groups[key][1]+=p*score;proposal_trace+=p*float(score@score)
    physical_trace=sum(float(score@score)/mass for mass,score in groups.values() if mass>0)
    return proposal_trace,physical_trace


def run(diagnostic,output):
    source=Path(diagnostic).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();manifest=dict(status='RUNNING',diagnostic=str(source),source_manifest_sha256=sha(source/'manifest.json'))
    write(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<900,'15minute verification stop')
    try:
        m=read(source/'manifest.json');check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['protocol']=='technician_broker_aliasing_diagnostic_v1' and m['updates']==0 and m['physical_rollouts']==0 and not m['test_opened'],'terminal read-only diagnostic')
        raw=Path(m['source']);rawv=Path(m['verification']);bound=0
        for directory,key in [(source,'output_bindings'),(raw,'output_bindings'),(rawv,'outputs')]:
            mm=read(directory/'manifest.json')
            for name,meta in mm[key].items():
                guard();check(sha(directory/name)==meta['sha256'] and (directory/name).stat().st_size==meta['bytes'],'binding '+name);bound+=1
        check(sha(raw/'manifest.json')==m['source_manifest_sha256'] and sha(rawv/'manifest.json')==m['verification_manifest_sha256'],'source references unchanged')
        for name,h in m['source_seal'].items():
            blob=subprocess.check_output(['git','show',f"{m['git']['commit']}:{name}"],cwd=ROOT);check(hashlib.sha256(blob).hexdigest()==h,'committed diagnostic dependency')
        spec=importlib.util.spec_from_file_location('primitive_alias_obs',ROOT/'scripts/verify_raw_reservation_budget_development.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        panel={};actual={}
        for r in csvrows(raw/'decisions.csv'):
            if r['method']=='independent_raw_ppo' and int(r['tick']) in range(0,48,3):
                key=int(r['train_seed']),int(r['seed']),int(r['tick']);obs=helper.observation(json.loads(r['state']),key[2]);check(key not in actual,'unique physical state panel');actual[key]=obs
                panel[key]=dict(train_seed=key[0],seed=key[1],tick=key[2],state_sha256=hashlib.sha256(r['state'].encode()).hexdigest(),observation_sha256=hashlib.sha256(obs['state'].tobytes()).hexdigest())
        stored=read(source/'panel.json');check(len(stored)==1536 and {tuple(r[k] for k in ('train_seed','seed','tick')) for r in stored}==set(panel),'complete panel')
        check(all(panel[r['train_seed'],r['seed'],r['tick']]==r for r in stored),'panel hashes')
        checkpoints=read(source/'checkpoint_bindings.json');check(len(checkpoints)==27 and {(r['train_seed'],r['steps']) for r in checkpoints}=={(t,s) for t in TRAINS for s in STEPS},'checkpoint grid')
        for r in checkpoints:check(sha(raw/r['path'])==r['sha256'],'checkpoint hash')
        rows=csvrows(source/'contexts.csv');lookup={(int(r['train_seed']),int(r['checkpoint_steps']),int(r['seed']),int(r['tick'])):r for r in rows}
        wanted={(t,step,s,tick) for t in TRAINS for step in STEPS for s in DEV for tick in range(0,48,3)};check(len(rows)==13824 and set(lookup)==wanted,'unique context grid')
        d=read(source/'summary.json');summaries={};checks={}
        for t in TRAINS:
            for step in STEPS:
                group=[lookup[t,step,s,tick] for s in DEV for tick in range(0,48,3)];p=sum(float(r['proposal_fisher_trace']) for r in group);f=sum(float(r['physical_fisher_trace']) for r in group);removed=sum(float(r['removed_fisher_trace']) for r in group)
                for r in group:near(float(r['removed_fisher_trace']),max(0,float(r['proposal_fisher_trace'])-float(r['physical_fisher_trace'])),'per-context difference')
                values=dict(contexts=512,proposal_fisher_trace_mean=p/512,physical_fisher_trace_mean=f/512,removed_fraction=removed/p if p>1e-20 else 0,
                    proposal_entropy_per_technician=np.mean([float(r['proposal_entropy_per_technician']) for r in group]),conflict_probability=np.mean([float(r['conflict_probability']) for r in group]))
                summaries[f'{t}:{step}']=values
                for k,v in values.items():near(v,d['summaries'][f'{t}:{step}'][k],'summary '+k)
            r=summaries[f'{t}:122880'];checks[str(t)]=dict(removed_actor_fraction=r['removed_fraction']>=.10,proposal_entropy=r['proposal_entropy_per_technician']>=.10,conflict_probability=r['conflict_probability']>=.05)
        passed=all(v for c in checks.values() for v in c.values());check(d['primary_checkpoint']==122880 and d['primary_checks']==checks and d['future_trial_screen_pass']==m['future_trial_screen_pass']==passed,'original engineering screen')
        physics=read(raw/'physics.json');quality=np.array(physics['restoration'])/np.array(physics['service_time']);ranks=np.zeros((4,2),int)
        for machine in range(4):
            for rank,j in enumerate(sorted(range(2),key=lambda j:(-quality[machine,j],j))):ranks[machine,j]=rank
        torch.set_num_threads(1);samples=[]
        for t in tqdm(TRAINS,desc='Direct parameter-score verification'):
            candidates=[key for key,o in actual.items() if key[0]==t and o['pair_mask'].all(1).sum()>=2];check(bool(candidates),'nontrivial eligible context exists');key=min(candidates);obs=actual[key]
            inputs=np.stack([np.concatenate([obs['state'],obs['technicians'][j],obs['edges'][:,j].flatten(),np.zeros(8),np.zeros(4),[j/2]]) for j in range(2)])
            for step in STEPS:
                guard();entry=next(r for r in checkpoints if r['train_seed']==t and r['steps']==step);data=torch.load(raw/entry['path'],map_location='cpu',weights_only=True)
                weights={k:v.double().requires_grad_() for k,v in data['state_dict'].items() if k.startswith('actor.')};check(sum(v.numel() for v in weights.values())==11141,'full shared actor')
                prop,phys=direct_fisher(inputs,weights,obs['pair_mask'],ranks);expected=lookup[t,step,key[1],key[2]]
                near(prop,float(expected['proposal_fisher_trace']),'direct proposal parameter score');near(phys,float(expected['physical_fisher_trace']),'direct grouped physical parameter score')
                samples.append(dict(train_seed=t,checkpoint_steps=step,seed=key[1],tick=key[2],proposal_trace=prop,physical_trace=phys))
        check(sha(source/'manifest.json')==manifest['source_manifest_sha256'],'diagnostic unchanged')
        result=dict(status='PASS',source=str(source),bindings_verified=bound,contexts_checked=13824,panel_states=1536,checkpoint_hashes=27,direct_parameter_contexts=27,direct_samples=samples,
            engineering_screen_pass=passed,primary_checks=checks,goal_pass=False,test_opened=False,
            limitations=['27 direct parameter-score contexts, not all13824 Jacobians independently recomputed.','Fixed selected greedy DEV panel, not on-policy return-gradient variance or cost improvement.'])
        write(out/'verification.json',result);shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'docs/technician_broker_aliasing_verification_plan.md',out/'protocol.md')
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});write(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',screen_pass=passed)))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');write(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.diagnostic,args.output)
