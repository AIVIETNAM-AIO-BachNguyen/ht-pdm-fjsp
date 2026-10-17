"""Frozen F64 actor-gradient diagnostic; no optimization or policy selection."""
import argparse,hashlib,importlib.util,json,shutil,subprocess,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_removal_experiment as e
from ht_pdm_fjsp.technician_broker_removal_policy import baseline_cache

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_broker_removal_gradient_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
TRAINS=tuple(range(330000000,330000003));STEPS=(0,122880,245760,368640,491520);KINDS=('state','conditional','proposal')


def reference():
    spec=importlib.util.spec_from_file_location('removal_independent',ROOT/'scripts/verify_broker_removal_development.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,m.helpers()


class DiagnosticEnv(e.ResourceEnv):
    def __init__(self,*args,physical_version=VERSION,**kwargs):
        self.physical_version=physical_version;super().__init__(*args,**kwargs)

    def reset(self):
        self.episode_index+=1;self.physical_seed=e.role_seed(self.seed,f'{self.physical_version}:N4:physical');self.env.reset(self.physical_seed);self.active=True;self.broker_totals=dict.fromkeys(('order_forward','order_reverse','conflicts','rejected_proposals'),0)
        self.population=hashlib.sha256(json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True).encode()).hexdigest();return self.observe()


def double_batch(observations):
    return {k:v.double() if v.is_floating_point() else v for k,v in e.learner.tensors(observations).items()}


def close(actual,expected,label,gradient=False):
    a=np.asarray(actual);b=np.asarray(expected);tolerance=1e-9+(1e-9 if gradient else 1e-10)*np.abs(b)
    e.check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and (np.abs(a-b)<=tolerance).all(),label)
    return float(np.max(np.abs(a-b)))


def numpy_vjp(observations,actions,advantage,weights,h,proposal=False):
    """Closed softmax/factor score and full shared MLP backprop, no torch."""
    x=[]
    for o in observations:
        for j in range(2):x.append(np.r_[o['state'],o['technicians'][j],o['edges'][:,j].ravel(),np.zeros(12),j/2])
    x=np.asarray(x,float);h1=np.tanh(x@weights['actor.0.weight'].T+weights['actor.0.bias']);h2=np.tanh(h1@weights['actor.2.weight'].T+weights['actor.2.bias']);lp=h.core_log_probs(observations,weights);p=np.exp(lp);dz=np.zeros_like(p)
    for row,(o,u) in enumerate(zip(observations,actions)):
        _,physical,_=h.independent_decode(u,o['pair_mask'],'broker_marginal_ppo',o['edges']);priority=h.priority_matrix(o,'broker_marginal_ppo')
        for i,token in enumerate(u if proposal else physical):
            score=-p[row,i].copy()
            if token:score[token]+=1
            elif proposal:score[0]+=1
            else:
                allowed=[0]+[int(m) for j,m in enumerate(physical) if m and (priority[int(m)-1,j]>priority[int(m)-1,i] or (priority[int(m)-1,j]==priority[int(m)-1,i] and j<i))]
                score[allowed]+=p[row,i,allowed]/p[row,i,allowed].sum()
            dz[row,i]=score*advantage[row,i]/(len(actions)*2)
    dz=dz.reshape(-1,5);w4=dz.T@h2;b4=dz.sum(0);d2=(dz@weights['actor.4.weight'])*(1-h2*h2);w2=d2.T@h1;b2=d2.sum(0);d0=(d2@weights['actor.2.weight'])*(1-h1*h1);w0=d0.T@x;b0=d0.sum(0)
    return np.concatenate([a.ravel() for a in (w0,b0,w2,b2,w4,b4)])


def trace_covariance(vectors):
    a=np.asarray(vectors,float);e.check(a.ndim==2 and len(a)>1 and np.isfinite(a).all(),'finite independent gradient samples');return float(np.square(a-a.mean(0)).sum()/(len(a)-1))


def metrics(gradients):
    variances={k:trace_covariance(v) for k,v in gradients.items()};e.check(all(v>1e-20 for v in variances.values()),'nondegenerate variances')
    means={k:v.mean(0) for k,v in gradients.items()}
    return dict(variance=variances,gain_vs_state=1-variances['conditional']/variances['state'],gain_vs_proposal=1-variances['conditional']/variances['proposal'],mean_gradient_norm={k:float(np.linalg.norm(v)) for k,v in means.items()},mean_difference_vs_state=float(np.linalg.norm(means['conditional']-means['state'])))


def primary(records):
    e.check(len(records)==15 and {(r['train_seed'],r['steps']) for r in records}=={(t,s) for t in TRAINS for s in STEPS},'all15fixed nodes');result={};checks={}
    for t in TRAINS:
        rr=[r for r in records if r['train_seed']==t];gain=float(np.mean([r['gain_vs_state'] for r in rr]));positive=sum(r['gain_vs_state']>0 for r in rr);result[str(t)]=dict(mean_gain=gain,positive_checkpoints=positive);checks[str(t)]=dict(material_mean_gain=gain>=.05,majority_positive=positive>=3)
    return result,checks


def replay_physics(directory,raw_seeds,arrays,observations,episode_metrics,h,guard,physical_version=VERSION):
    episodes={};count=0;policy={}
    for row in h.rows(directory/'policy.csv'):policy[int(row['episode']),int(row['tick'])]=row
    e.check(len(policy)==48*len(raw_seeds),'complete policy log')
    for row in h.rows(directory/'physical.csv'):
        if count%1000==0:guard()
        i,tick=int(row['episode']),int(row['tick']);s=json.loads(row['state']);pairs=tuple(map(tuple,json.loads(row['action'])));physical=h.seed(raw_seeds[i],f'{physical_version}:N4:physical')
        if i not in episodes:
            initial,events,pop=h.initial_events(physical);episodes[i]=dict(state=initial,events=events,pop=pop,tick=0,metrics={},reward=np.zeros(48))
        ep=episodes[i];e.check(ep['state']==s and ep['tick']==tick and json.loads(row['events'])==ep['events'][tick] and int(row['physical_seed'])==physical,'fresh seeded state/event chain')
        if tick<48:
            o=h.observation(s,tick);index=i*48+tick
            for name,value in o.items():e.check(np.array_equal(value,observations[name][index]),'independent observation '+name)
            u=arrays['actions'][index];expected,_,diag=h.independent_decode(u,o['pair_mask'],'broker_marginal_ppo',o['edges']);pr=policy[i,tick];e.check(expected==pairs and json.loads(pr['action'])==u.tolist() and json.loads(pr['diagnostics'])==diag and hashlib.sha256(o['state'].tobytes()).hexdigest()==pr['observation_sha256'],'policy mapping/join')
        else:
            score=np.array([[h.urgency(s,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);h.near(sum(score[m,j] for m,j in pairs),h.top_two_optimum(score,h.support(s)),'common tail')
        following,parts=h.primitive_step(s,pairs,ep['events'][tick]);e.check(following==json.loads(row['following']),'primitive transition');h.near(parts['objective'],row['cost'],'primitive cost');ep['reward'][min(tick,47)]-=parts['objective']/20
        for k,v in parts.items():ep['metrics'][k]=ep['metrics'].get(k,0)+v
        ep['state']=following;ep['tick']+=1;count+=1
    for i,ep in episodes.items():
        e.check(ep['tick']==60 and ep['pop']==episode_metrics[i]['population_sha256'],'terminal tape')
        for k,v in ep['metrics'].items():h.near(v,episode_metrics[i][k],'episode metric '+k)
        close(arrays['rewards'].reshape(-1,48)[i],ep['reward'],'independent rewards');close(arrays['returns'].reshape(-1,48)[i],np.cumsum(ep['reward'][::-1])[::-1],'independent returns')
    e.check(count==len(raw_seeds)*60 and len(episodes)==len(raw_seeds),'complete physical grid');return count


def collect(model,raw_seeds,action_seed,directory,guard,physical_version=VERSION):
    """One fixed model, shared trajectories for all three estimators."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);r,h=reference();seal=e.learner.parameter_digest(model);e.check(next(model.actor.parameters()).dtype==torch.float64,'frozen F64 actor')
    physical=e.Journal(directory/'physical.csv');policy=e.Journal(directory/'policy.csv');envs=[DiagnosticEnv(raw,4,e.CORE[0],journal=e.old.Tags(physical,dict(episode=i)),physical_version=physical_version) for i,raw in enumerate(raw_seeds)];obs=[env.reset() for env in envs];rng=torch.Generator().manual_seed(action_seed);histories=[[] for _ in envs];actions=[[] for _ in envs];rewards=[[] for _ in envs];episode_metrics={}
    try:
        for tick in range(48):
            guard();batch=double_batch(obs);u,_,_=model.sample(batch,rng)
            following=[]
            for i,env in enumerate(envs):
                histories[i].append(obs[i]);action=u[i].numpy().copy();actions[i].append(action);pairs,diag=env.resolve(action);policy.add(dict(episode=i,tick=tick,observation_sha256=hashlib.sha256(obs[i]['state'].tobytes()).hexdigest(),action=json.dumps(action.tolist()),pairs=json.dumps(pairs),diagnostics=json.dumps(diag)));o,reward,done,info=env.step(action);rewards[i].append(reward);following.append(o)
                if tick==47:e.check(done,'H48');episode_metrics[i]=info['episode_metrics']
            obs=following
    finally:physical.close();policy.close()
    flat=[o for episode in histories for o in episode];observations={k:np.stack([o[k] for o in flat]) for k in flat[0]};u=np.asarray(actions).reshape(-1,2);reward=np.asarray(rewards);returns=np.cumsum(reward[:,::-1],axis=1)[:,::-1].copy().ravel();batch=double_batch(flat);weights={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    with torch.no_grad():
        logs,_,value=model.evaluate(batch,torch.tensor(u));cache=baseline_cache(model,batch,torch.tensor(u),value)
    dense=r.dense_baselines(flat,u,weights,h);errors={};errors['logs']=close(logs.numpy(),dense['factors'],'dense physical factors');errors['value']=close(value.numpy(),h.mlp(observations['state'].astype(float),weights,'critic').ravel()*115.2,'NumPy state V');errors['actual']=close(cache['actual_values'].numpy(),dense['actual'],'dense actual removal');errors['conditional']=close(cache['mixed_values'].numpy(),dense['mixed'],'dense conditional removal')
    arrays=dict(actions=u,rewards=reward.ravel(),returns=returns,value=value.numpy(),actual=cache['actual_values'].numpy(),conditional=cache['mixed_values'].numpy(),logs=logs.numpy());physical_count=replay_physics(directory,raw_seeds,arrays,observations,episode_metrics,h,guard,physical_version)
    replay_rng=torch.Generator().manual_seed(action_seed);lp=dense['logp'].reshape(len(raw_seeds),48,2,5)
    for tick in range(48):
        expected=torch.stack([torch.multinomial(torch.tensor(np.exp(lp[:,tick,j])),1,generator=replay_rng).squeeze(-1) for j in range(2)],-1).numpy();e.check(np.array_equal(expected,u.reshape(-1,48,2)[:,tick]),'independent F64 stochastic draw replay')
    gradients={k:[] for k in KINDS};gradient_errors={k:0. for k in KINDS};parameters=tuple(model.actor.parameters())
    for i,episode in enumerate(tqdm(histories,desc='Paired full actor gradients',leave=False)):
        guard();sl=slice(i*48,(i+1)*48);b=double_batch(episode);a=torch.tensor(u[sl]);factor,_,_=model.evaluate(b,a);proposal=model.proposal_log_probs(b)[0].gather(-1,a[...,None]).squeeze(-1);g=returns[sl,None];advantages=dict(state=np.repeat(g-arrays['value'][sl,None],2,axis=1),conditional=g-arrays['conditional'][sl],proposal=g-arrays['actual'][sl])
        for kind in KINDS:
            score=proposal if kind=='proposal' else factor;objective=(score*torch.tensor(advantages[kind])).mean();vector=torch.cat([v.flatten() for v in torch.autograd.grad(objective,parameters,retain_graph=kind=='state')]).numpy();expected=numpy_vjp(episode,u[sl],advantages[kind],weights,h,kind=='proposal');gradient_errors[kind]=max(gradient_errors[kind],close(vector,expected,'closed MLP VJP '+kind,True));gradients[kind].append(vector)
    gradients={k:np.stack(v) for k,v in gradients.items()};e.check(e.learner.parameter_digest(model)==seal,'all frozen tensors unchanged');np.savez_compressed(directory/'observations.npz',**observations);np.savez_compressed(directory/'arrays.npz',**arrays);np.savez_compressed(directory/'gradients.npz',**gradients);e.write_csv(directory/'episodes.csv',[dict(episode=i,**episode_metrics[i]) for i in range(len(raw_seeds))]);result=dict(status='PASS',episodes=len(raw_seeds),physical_transitions=physical_count,policy_decisions=48*len(raw_seeds),optimizer_updates=0,parameter_sha256=seal,baseline_max_errors=errors,gradient_max_errors=gradient_errors,scalar_residual_ratio=float(np.var(returns[:,None]-arrays['conditional'])/np.var(returns-arrays['value'])))
    if len(raw_seeds)>1:result.update(metrics(gradients))
    e.write_json(directory/'summary.json',result);return result


def seed_audit(h):
    registry=e.read(REGISTRY);raw=[*registry['physical_seeds'],*registry['action_bases'],registry['analysis_seed'],*registry['engineering'].values()];own=h.sha(REGISTRY);historical=set();files={}
    def ints(x):
        if isinstance(x,dict):return set.union(set(),*(ints(v) for v in x.values()))
        if isinstance(x,list):return set.union(set(),*(ints(v) for v in x))
        return {x} if isinstance(x,int) and not isinstance(x,bool) else set()
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp/configs').glob('*seed_registry.json')]:
        digest=h.sha(p)
        if digest!=own and digest not in files:historical|=ints(h.read(p));files[digest]=str(p)
    derived=[h.seed(s,f'{VERSION}:N4:physical') for s in (*registry['physical_seeds'],registry['engineering']['physical_seed'])]+[h.seed(s,VERSION+':actions') for s in (*registry['action_bases'],registry['engineering']['action_seed'])]+[h.seed(registry['analysis_seed'],VERSION+':analysis')]
    e.check(len(raw)==len(set(raw))==146 and len(derived)==len(set(derived))==146 and not set(raw)&historical and not set(derived)&(set(raw)|historical),'fresh diagnostic streams')
    return dict(status='PASS',raw=146,derived=146,historical_registries=files,confirmation_states_constructed=False,limitation='Historical raw registries checked; all historical derived namespaces not exhaustively reconstructed.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);r,h=reference();torch.set_num_threads(1);start=time.monotonic();sm=h.read(source/'manifest.json');vm=h.read(verification/'manifest.json');v=h.read(verification/'verification.json');manifest=dict(status='RUNNING',protocol=VERSION,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),verification_manifest_sha256=h.sha(verification/'manifest.json'),goal_pass=False,test_opened=False,optimizer_updates=0)
    e.write_json(out/'manifest.json',manifest)
    def guard():e.check(time.monotonic()-start<3600,'60minute hardstop')
    def bindings():
        e.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'] and h.sha(verification/'manifest.json')==manifest['verification_manifest_sha256'],'source manifests unchanged')
        for name,meta in sm['output_bindings'].items():guard();e.check(h.sha(source/name)==meta['sha256'],'source binding '+name)
    try:
        e.check(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean pushed source');e.check(sm['status']=='COMPLETED' and sm['git']['commit']==r.COMMIT and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source_manifest_sha256']==manifest['source_manifest_sha256'] and not sm['development_screen_pass'] and not sm['test_opened'],'verified negative source; no scientific promotion');bindings()
        for name,meta in vm['outputs'].items():e.check(h.sha(verification/name)==meta['sha256'],'verifier binding '+name)
        audit=seed_audit(h);e.write_json(out/'seed_audit.json',audit);registry=h.read(REGISTRY);shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');shutil.copy2(__file__,out/'diagnostic.py');e.write_json(out/'resolved_config.json',dict(protocol=VERSION,source_checkpoint_grid=[dict(train_seed=t,steps=s) for t in TRAINS for s in STEPS],episodes_per_group=128,horizon=48,tail=12,machines=4,technicians=2,dtype='float64 promoted stored weights',optimizer_updates=0,gradient_kinds=KINDS,material_gain=.05,majority_positive=3,max_seconds=3600));seals={p.name:h.sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'};config=e.settings();records=[];code_seals={str(p):h.sha(p) for p in (Path(__file__),PLAN,REGISTRY,ROOT/'scripts/verify_broker_removal_development.py',ROOT/'scripts/verify_broker_marginal_development.py')}
        for name,digest in sm['source_seal'].items():
            if name.startswith('source_snapshot/'):
                path=Path(__file__).parent/Path(name).name;e.check(h.sha(path)==digest,'frozen production module '+name);code_seals[str(path)]=digest
        e.write_json(out/'code_seals.json',code_seals)
        for index,(t,step) in enumerate(tqdm([(t,s) for t in TRAINS for s in STEPS],desc='15frozen FULL diagnostic groups')):
            guard();checkpoint=source/f'N4/{e.CORE[0]}/{t}/checkpoints/step_{step:09d}.pt';model=e.load_model(checkpoint,e.CORE[0],config,t,step).double();model.critic.requires_grad_(False);model.removal.requires_grad_(False);directory=out/f'{t}_{step}';action_seed=h.seed(registry['action_bases'][index],VERSION+':actions');record=collect(model,registry['physical_seeds'],action_seed,directory,guard);record.update(train_seed=t,steps=step,source_checkpoint=str(checkpoint.relative_to(source)),source_checkpoint_sha256=h.sha(checkpoint),action_seed=action_seed)
            if step==0:
                record['raw_step0_gain']=record['gain_vs_state'];e.check(abs(record['raw_step0_gain'])<1e-10,'initial copy gradient variance invariance');record['gain_vs_state']=0.
            records.append(record);e.write_json(out/'records.partial.json',records)
        means,checks=primary(records);summary=dict(classification='FROZEN_ACTOR_GRADIENT_DIAGNOSTIC_ONLY',records=records,replicas=means,checks=checks,mechanism_screen_pass=all(ok for row in checks.values() for ok in row.values()),source_cost_verdict_overridden=False,goal_pass=False,test_opened=False,optimizer_updates=0,episodes=1920,policy_decisions=92160,physical_transitions=115200,limitation='Unclipped unnormalized frozen F64 reward gradients; diagnostic point estimates, no scientific CI or PPO update/cost benefit claim. Source negative verdict remains unchanged.')
        e.check(sum(r['episodes'] for r in records)==1920 and sum(r['physical_transitions'] for r in records)==115200 and sum(r['policy_decisions'] for r in records)==92160,'full fixed budget');bindings()
        for name,digest in seals.items():e.check(h.sha(out/name)==digest,'diagnostic source seal')
        for path,digest in code_seals.items():e.check(h.sha(Path(path))==digest,'live source unchanged')
        e.write_json(out/'summary.json',summary);manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,mechanism_screen_pass=summary['mechanism_screen_pass'],output_bindings={str(p.relative_to(out)):dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});e.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),mechanism_screen_pass=summary['mechanism_screen_pass'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);e.write_json(out/'manifest.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
