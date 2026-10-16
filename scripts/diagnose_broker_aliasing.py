"""Read-only conditional fixed-state actor-score diagnostic; no learning or rollout."""
import argparse,csv,hashlib,importlib.util,itertools,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
VERSION='technician_broker_aliasing_diagnostic_v1'
SOURCE_VERSION='technician_raw_reservation_budget_v1'
SOURCE_COMMIT='4ade42381681e073efc8a2c7dc9efb43640a3809'
TRAINS=tuple(range(280000000,280000003));DEV=tuple(range(280100000,280100032));STEPS=tuple(range(0,983041,122880))
METHOD='independent_raw_ppo'


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def load_helper():
    p=ROOT/'scripts/verify_raw_reservation_budget_development.py';spec=importlib.util.spec_from_file_location('alias_observation',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def decode(tokens,ranks):
    outcome=[0]*len(tokens)
    for token in set(tokens)-{0}:
        group=[j for j,t in enumerate(tokens) if t==token];j=min(group,key=lambda j:ranks[token-1,j]);outcome[j]=token
    return tuple(outcome)


def physical_score(logp,matching,ranks,legal):
    k,width=logp.shape;p=np.exp(logp);winners=[(t-1,j) for j,t in enumerate(matching) if t];score=-p.copy();logmass=0.
    check(len({m for m,j in winners})==len(winners) and all(legal[m,j] for m,j in winners),'feasible matching')
    for j,token in enumerate(matching):
        if token:logmass+=logp[j,token];score[j,token]+=1
        else:
            allowed=[0]+[m+1 for m,w in winners if legal[m,j] and ranks[m,w]<ranks[m,j]]
            factor=np.logaddexp.reduce(logp[j,allowed]);logmass+=factor;score[j,allowed]+=np.exp(logp[j,allowed]-factor)
    return float(logmass),score.ravel()


def noise_metrics(logp,gram,ranks,legal):
    k,width=logp.shape;n=width-1;p=np.exp(logp);groups={};proposal_trace=0.;mean_score=np.zeros(k*width)
    support=[[0]+[m+1 for m in range(n) if legal[m,j]] for j in range(k)]
    for tokens in itertools.product(*support):
        mass=float(np.exp(sum(logp[j,t] for j,t in enumerate(tokens))));score=-p.copy();score[np.arange(k),tokens]+=1;score=score.ravel();outcome=decode(tokens,ranks)
        if outcome not in groups:groups[outcome]=[0.,np.zeros_like(score)]
        groups[outcome][0]+=mass;groups[outcome][1]+=mass*score
        proposal_trace+=mass*float(score@gram@score);mean_score+=mass*score
    physical_trace=0.;mass_sum=0.
    for matching,(mass,weighted_score) in groups.items():
        logmass,score=physical_score(logp,matching,ranks,legal);q=float(np.exp(logmass));mass_sum+=q
        check(np.allclose(q,mass,atol=2e-10,rtol=2e-10),'closed mass vs exhaustive sum')
        check(np.allclose(q*score,weighted_score,atol=2e-10,rtol=2e-10),'closed score vs posterior score')
        physical_trace+=q*float(score@gram@score)
    check(abs(mass_sum-1)<=2e-10 and np.abs(mean_score).max()<=2e-10,'normalization/score identity')
    removed=proposal_trace-physical_trace
    check(removed>=-2e-10*(1+abs(proposal_trace)) and min(proposal_trace,physical_trace)>=-2e-10,'variance contraction')
    finite=np.isfinite(logp);entropy=-float((p[finite]*logp[finite]).sum())/k
    check(k==2,'diagnostic conflict expression is K2')
    return dict(proposal_fisher_trace=proposal_trace,physical_fisher_trace=physical_trace,removed_fisher_trace=max(0.,removed),
        removed_fraction=max(0.,removed)/proposal_trace if proposal_trace>1e-20 else 0.,
        proposal_entropy_per_technician=entropy,conflict_probability=float((p[0,1:]*p[1,1:]).sum()),
        physical_actions=len(groups),proposal_actions=math.prod(map(len,support)))


def actor_logits_gram(inputs,weights):
    array=torch.as_tensor(inputs,dtype=torch.float64)
    for i in (0,2,4):
        array=array@weights[f'actor.{i}.weight'].T+weights[f'actor.{i}.bias']
        if i!=4:array=torch.tanh(array)
    jacobian=[];flat=array.flatten();params=tuple(weights.values())
    for i,output in enumerate(flat):
        grad=torch.autograd.grad(output,params,retain_graph=i<len(flat)-1)
        jacobian.append(torch.cat([g.flatten() for g in grad]).detach())
    jacobian=torch.stack(jacobian)
    return array.detach().numpy(),(jacobian@jacobian.T).numpy()


def inputs_for(obs):
    return np.stack([np.concatenate([obs['state'],obs['technicians'][j],obs['edges'][:,j].ravel(),np.zeros(12),[j/2]]) for j in range(2)])


def summarize(records):
    expected={(t,step,s,tick) for t in TRAINS for step in STEPS for s in DEV for tick in range(0,48,3)}
    check(len(records)==len(expected) and {(r['train_seed'],r['checkpoint_steps'],r['seed'],r['tick']) for r in records}==expected,'unique full checkpoint/context grid')
    summaries={}
    for t in TRAINS:
        for step in STEPS:
            group=[r for r in records if r['train_seed']==t and r['checkpoint_steps']==step]
            check(len(group)==512,'complete context grid');proposal=sum(r['proposal_fisher_trace'] for r in group);removed=sum(r['removed_fisher_trace'] for r in group)
            summaries[f'{t}:{step}']=dict(contexts=512,proposal_fisher_trace_mean=proposal/512,
                physical_fisher_trace_mean=sum(r['physical_fisher_trace'] for r in group)/512,removed_fraction=removed/proposal if proposal>1e-20 else 0.,
                proposal_entropy_per_technician=float(np.mean([r['proposal_entropy_per_technician'] for r in group])),conflict_probability=float(np.mean([r['conflict_probability'] for r in group])))
    checks={}
    for t in TRAINS:
        r=summaries[f'{t}:122880']
        checks[str(t)]=dict(removed_actor_fraction=r['removed_fraction']>=.10,proposal_entropy=r['proposal_entropy_per_technician']>=.10,conflict_probability=r['conflict_probability']>=.05)
    return dict(classification='FIXED_DEV_STATE_SCORE_DIAGNOSTIC',summaries=summaries,primary_checkpoint=122880,
        primary_checks=checks,future_trial_screen_pass=all(v for c in checks.values() for v in c.values()),goal_pass=False,test_opened=False,
        limitation='Conditional unit-weight actor-score Fisher on selected greedy DEV states; not on-policy return-gradient variance, clipping-gradient variance, cost evidence or scientific CI.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    manifest=dict(status='RUNNING',protocol=VERSION,source=str(source),verification=str(verification),source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,updates=0,physical_rollouts=0)
    manifest['git']=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()))
    manifest['source_seal']={name:sha(ROOT/name) for name in ('scripts/diagnose_broker_aliasing.py','scripts/broker_marginal_algebra.py','scripts/verify_raw_reservation_budget_development.py',f'docs/{VERSION}_plan.md')}
    write(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<900,'15minute diagnostic stop')
    def bindings():
        m=read(source/'manifest.json');vm=read(verification/'manifest.json')
        check(sha(source/'manifest.json')==manifest['source_manifest_sha256'] and sha(verification/'manifest.json')==manifest['verification_manifest_sha256'],'input manifests unchanged')
        for name,meta in m['output_bindings'].items():
            guard();check(sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'source binding '+name)
        for name,meta in vm['outputs'].items():check(sha(verification/name)==meta['sha256'],'verification binding '+name)
    try:
        check(not manifest['git']['dirty'],'clean pushed diagnostic code required')
        m=read(source/'manifest.json');v=read(verification/'verification.json');vm=read(verification/'manifest.json')
        check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==SOURCE_COMMIT and not m['git']['dirty'] and m['protocol']==SOURCE_VERSION,'frozen source terminal')
        check(v['status']=='PASS' and vm['status']=='COMPLETED' and v['source']==str(source) and v['source_manifest_sha256']==manifest['source_manifest_sha256'],'independent source verification')
        check(not v['original_development_pass'] and not m['development_screen_pass'] and not m['test_opened'],'conditional diagnostic requires rejected DEV eligibility')
        lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines();check(not any('from multiprocessing.spawn import spawn_main' in x for x in lines),'no active training workers')
        torch.set_num_threads(1);bindings();helper=load_helper();contexts={};panel=[]
        with (source/'decisions.csv').open() as f:
            for r in csv.DictReader(f):
                if r['method']!=METHOD or int(r['tick']) not in range(0,48,3):continue
                t,s,tick=int(r['train_seed']),int(r['seed']),int(r['tick']);check(t in TRAINS and s in DEV,'source DEV state panel');key=t,s,tick
                check(key not in contexts,'duplicate panel state');state=json.loads(r['state']);obs=helper.observation(state,tick);contexts[key]=obs
                panel.append(dict(train_seed=t,seed=s,tick=tick,state_sha256=hashlib.sha256(r['state'].encode()).hexdigest(),observation_sha256=hashlib.sha256(obs['state'].tobytes()).hexdigest()))
        check(set(contexts)=={(t,s,tick) for t in TRAINS for s in DEV for tick in range(0,48,3)},'1536 fixed contexts')
        physics=read(source/'physics.json');quality=np.array(physics['restoration'])/np.array(physics['service_time']);ranks=np.zeros((4,2),int)
        for machine in range(4):
            order=sorted(range(2),key=lambda j:(-quality[machine,j],j))
            for rank,j in enumerate(order):ranks[machine,j]=rank
        records=[];checkpoint_bindings=[]
        with tqdm(total=13824,desc='Frozen actor score contexts',mininterval=1) as progress:
            for t in TRAINS:
                for step in STEPS:
                    path=source/'N4'/METHOD/str(t)/'checkpoints'/f'step_{step:09d}.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
                    check(all(payload[k]==value for k,value in dict(protocol=SOURCE_VERSION,algorithm=METHOD,train_seed=t,steps=step).items()),'checkpoint identity')
                    checkpoint_bindings.append(dict(train_seed=t,steps=step,path=str(path.relative_to(source)),sha256=sha(path)))
                    weights={k:x.detach().double().requires_grad_() for k,x in payload['state_dict'].items() if k.startswith('actor.')}
                    check(sum(x.numel() for x in weights.values())==11141,'full shared actor parameter count')
                    for s in DEV:
                        for tick in range(0,48,3):
                            guard();obs=contexts[t,s,tick];z,gram=actor_logits_gram(inputs_for(obs),weights);support=np.c_[np.ones(2,bool),obs['pair_mask'].T]
                            masked=np.where(support,z,-np.inf);logp=masked-np.logaddexp.reduce(masked,axis=1)[:,None]
                            metrics=noise_metrics(logp,gram,ranks,obs['pair_mask']);records.append(dict(train_seed=t,checkpoint_steps=step,seed=s,tick=tick,**metrics));progress.update()
        result=summarize(records);write(out/'summary.json',result);write(out/'panel.json',panel);write(out/'checkpoint_bindings.json',checkpoint_bindings)
        with (out/'contexts.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(records[0]));writer.writeheader();writer.writerows(records)
        shutil.copy2(__file__,out/'diagnostic.py');shutil.copy2(ROOT/'docs'/f'{VERSION}_plan.md',out/'protocol.md')
        for name in ('broker_marginal_algebra.py','verify_raw_reservation_budget_development.py'):shutil.copy2(ROOT/'scripts'/name,out/name)
        bindings()
        check(all(sha(ROOT/name)==h for name,h in manifest['source_seal'].items()),'diagnostic code unchanged')
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,contexts=len(contexts),checkpoint_contexts=len(records),checkpoints=27,
            future_trial_screen_pass=result['future_trial_screen_pass'],output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        write(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',screen_pass=result['future_trial_screen_pass'])));return result
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)

if __name__=='__main__':main()
