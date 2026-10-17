"""Independent dense-category shared-parameter gradient replay, read-only."""
import argparse,hashlib,importlib.util,itertools,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];VERSION='technician_broker_removal_gradient_diagnostic_v1';COMMIT='b1948c33f7041fc8fe825d84f1738c3719aeb4e4';TRAINS=tuple(range(330000000,330000003));STEPS=(0,122880,245760,368640,491520);KINDS=('state','conditional','proposal')
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def close(a,b,label,gradient=False):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and (np.abs(a-b)<=1e-9+(1e-9 if gradient else 1e-10)*abs(b)).all(),label
    return float(np.max(abs(a-b)))


def dense_gradients(observations,actions,returns,value,actual,conditional,weights,h,manual_logp):
    """Different estimator expression: enumerate preimages and detach other p."""
    packed={k:np.stack([o[k] for o in observations]) for k in observations[0]};lp,parameters=manual_logp(packed,weights,torch.float64,True);p=lp.exp();latent=np.array(list(itertools.product(range(5),repeat=2)));table=np.array([h.independent_decode(u,np.ones((4,2),bool),'broker_marginal_ppo',observations[0]['edges'])[1] for u in latent]);physical=np.array([h.independent_decode(u,o['pair_mask'],'broker_marginal_ppo',o['edges'])[1] for u,o in zip(actions,observations)]);same=torch.tensor((table[None]==physical[:,None]).all(-1),dtype=torch.float64);terms=[]
    for i in range(2):terms.append(((p[:,i,latent[:,i]]*p[:,1-i,latent[:,1-i]].detach())*same).sum(-1).log())
    # These logs have value log q, but each differentiates ONLY one agent's p.
    factor_score=torch.stack(terms,-1);proposal=lp.gather(-1,torch.tensor(actions)[...,None]).squeeze(-1);g=returns[:,None];advantages=dict(state=np.repeat(g-value[:,None],2,axis=1),conditional=g-conditional,proposal=g-actual);result={}
    for kind in KINDS:
        score=proposal if kind=='proposal' else factor_score;objective=(score*torch.tensor(advantages[kind],dtype=torch.float64)).mean();result[kind]=torch.cat([g.flatten() for g in torch.autograd.grad(objective,tuple(parameters.values()),retain_graph=kind!='proposal')]).numpy()
    return result


def group_physics(directory,observations,arrays,seeds,h,guard,physical_version=VERSION):
    size=len(seeds);episodes={};count=0;policy=list(h.rows(directory/'policy.csv'));assert len(policy)==48*size;policies={(int(r['episode']),int(r['tick'])):r for r in policy};assert len(policies)==48*size
    for row in h.rows(directory/'physical.csv'):
        if count%1000==0:guard()
        i,tick=int(row['episode']),int(row['tick']);physical=h.seed(seeds[i],f'{physical_version}:N4:physical');s=json.loads(row['state']);pairs=tuple(map(tuple,json.loads(row['action'])))
        if i not in episodes:
            initial,events,pop=h.initial_events(physical);episodes[i]=dict(state=initial,events=events,pop=pop,tick=0,metrics={},reward=np.zeros(48))
        ep=episodes[i];assert ep['state']==s and ep['tick']==tick and json.loads(row['events'])==ep['events'][tick] and int(row['physical_seed'])==physical
        if tick<48:
            index=i*48+tick;o=h.observation(s,tick)
            for name,v in o.items():assert np.array_equal(v,observations[name][index]),name
            pr=policies[i,tick];u=arrays['actions'][index];decoded,_,diag=h.independent_decode(u,o['pair_mask'],'broker_marginal_ppo',o['edges']);assert decoded==pairs and json.loads(pr['action'])==u.tolist() and json.loads(pr['diagnostics'])==diag and hashlib.sha256(o['state'].tobytes()).hexdigest()==pr['observation_sha256']
        else:
            score=np.array([[h.urgency(s,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);h.near(sum(score[m,j] for m,j in pairs),h.top_two_optimum(score,h.support(s)),'tail')
        following,parts=h.primitive_step(s,pairs,ep['events'][tick]);assert following==json.loads(row['following']);h.near(parts['objective'],row['cost'],'physicscost');ep['reward'][min(tick,47)]-=parts['objective']/20
        for k,v in parts.items():ep['metrics'][k]=ep['metrics'].get(k,0)+v
        ep['state']=following;ep['tick']+=1;count+=1
    rows=list(h.rows(directory/'episodes.csv'));assert len(rows)==size;metrics={int(r['episode']):r for r in rows};assert len(metrics)==size
    for i,ep in episodes.items():
        assert ep['tick']==60 and ep['pop']==metrics[i]['population_sha256']
        for k,v in ep['metrics'].items():h.near(v,metrics[i][k],'episode metric')
        close(arrays['rewards'].reshape(size,48)[i],ep['reward'],'rewards');close(arrays['returns'].reshape(size,48)[i],np.cumsum(ep['reward'][::-1])[::-1],'returns')
    assert count==size*60 and len(episodes)==size;return count


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);r=module('removal_dense_reference','verify_broker_removal_development.py');h=r.helpers();entropy=module('manual_parameter_reference','verify_broker_entropy_alignment.py');torch.set_num_threads(1);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def guard():assert time.monotonic()-start<1800,'30minute audit hardstop'
    m=h.read(source/'manifest.json')
    def bindings():
        assert h.sha(source/'manifest.json')==manifest['source_manifest_sha256']
        for name,meta in m['output_bindings'].items():guard();assert h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],name
    try:
        assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git_commit']==COMMIT and m['protocol']==VERSION and m['optimizer_updates']==0 and not m['test_opened'];bindings();s=h.read(source/'summary.json');registry=h.read(source/'seed_registry.json');assert registry['physical_seeds']==list(range(340100000,340100128)) and registry['action_bases']==list(range(340000000,340000015));assert len(s['records'])==15 and {(v['train_seed'],v['steps']) for v in s['records']}=={(t,step) for t in TRAINS for step in STEPS};records=[];transition_count=0
        for index,(t,step) in enumerate(tqdm([(t,step) for t in TRAINS for step in STEPS],desc='15independent dense gradient groups')):
            guard();record=next(v for v in s['records'] if v['train_seed']==t and v['steps']==step);directory=source/f'{t}_{step}';checkpoint=Path(m['source'])/record['source_checkpoint'];assert h.sha(checkpoint)==record['source_checkpoint_sha256'];payload=torch.load(checkpoint,map_location='cpu',weights_only=True);assert payload['steps']==step and payload['train_seed']==t and payload['algorithm']=='removal_rb_ppo';weights={k:v.numpy().astype(float) for k,v in payload['state_dict'].items()};digest=hashlib.sha256()
            for k,v in weights.items():digest.update(k.encode());digest.update(v.tobytes())
            assert digest.hexdigest()==record['parameter_sha256'];arrays={};observations={};gradients={}
            for filename,target in (('arrays.npz',arrays),('observations.npz',observations),('gradients.npz',gradients)):
                with np.load(directory/filename,allow_pickle=False) as f:target.update({k:f[k] for k in f.files})
            assert all(v.shape==(128,11141) for v in gradients.values()) and set(gradients)==set(KINDS);transition_count+=group_physics(directory,observations,arrays,registry['physical_seeds'],h,guard);obs=[{k:v[i] for k,v in observations.items()} for i in range(6144)];dense=r.dense_baselines(obs,arrays['actions'],weights,h);value=h.mlp(observations['state'].astype(float),weights,'critic').ravel()*115.2;close(arrays['value'],value,'stateV');close(arrays['actual'],dense['actual'],'actual baseline');close(arrays['conditional'],dense['mixed'],'conditional baseline');close(arrays['logs'],dense['factors'],'logs');rng=torch.Generator().manual_seed(h.seed(registry['action_bases'][index],VERSION+':actions'));assert record['action_seed']==rng.initial_seed();lp=dense['logp'].reshape(128,48,2,5)
            for tick in range(48):
                draws=torch.stack([torch.multinomial(torch.tensor(np.exp(lp[:,tick,j])),1,generator=rng).squeeze(-1) for j in range(2)],-1).numpy();assert np.array_equal(draws,arrays['actions'].reshape(128,48,2)[:,tick])
            errors={k:0. for k in KINDS}
            for episode in tqdm(range(128),desc='384full gradient vectors',leave=False):
                guard();sl=slice(episode*48,(episode+1)*48);reference=dense_gradients(obs[sl],arrays['actions'][sl],arrays['returns'][sl],value[sl],dense['actual'][sl],dense['mixed'][sl],weights,h,entropy.manual_logp)
                for kind in KINDS:errors[kind]=max(errors[kind],close(gradients[kind][episode],reference[kind],kind+' dense VJP',True))
            means={k:v.mean(0) for k,v in gradients.items()};variances={k:float(sum(np.dot(v-means[k],v-means[k]) for v in vectors)/127) for k,vectors in gradients.items()};assert all(v>1e-20 and np.isfinite(v) for v in variances.values())
            for k,v in variances.items():h.near(v,record['variance'][k],'tracecov '+k)
            for k,v in means.items():h.near(np.linalg.norm(v),record['mean_gradient_norm'][k],'mean norm')
            h.near(np.linalg.norm(means['conditional']-means['state']),record['mean_difference_vs_state'],'mean gradient difference');h.near(np.var(arrays['returns'][:,None]-dense['mixed'])/np.var(arrays['returns']-value),record['scalar_residual_ratio'],'scalar variance ratio')
            gain=1-variances['conditional']/variances['state'];raw_gain=gain
            if step==0:assert abs(gain)<1e-10;h.near(gain,record['raw_step0_gain'],'step0 raw');gain=0.
            h.near(gain,record['gain_vs_state'],'primary gain');h.near(1-variances['conditional']/variances['proposal'],record['gain_vs_proposal'],'RBgain');records.append(dict(train_seed=t,steps=step,gain_vs_state=gain,raw_gain=raw_gain,variance=variances,gradient_max_errors=errors))
        checks={};replicas={}
        for t in TRAINS:
            rr=[r for r in records if r['train_seed']==t];gain=float(np.mean([r['gain_vs_state'] for r in rr]));positive=sum(r['gain_vs_state']>0 for r in rr);replicas[str(t)]=dict(mean_gain=gain,positive_checkpoints=positive);checks[str(t)]=dict(material_mean_gain=gain>=.05,majority_positive=positive>=3);h.near(gain,s['replicas'][str(t)]['mean_gain'],'replica mean');assert positive==s['replicas'][str(t)]['positive_checkpoints']
        passed=all(v for row in checks.values() for v in row.values());assert checks==s['checks'] and passed==s['mechanism_screen_pass']==m['mechanism_screen_pass'] and transition_count==115200 and s['episodes']==1920 and s['policy_decisions']==92160 and s['optimizer_updates']==0 and not s['source_cost_verdict_overridden'] and not s['goal_pass'];bindings();result=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],records=records,replicas=replicas,checks=checks,mechanism_screen_pass=passed,gradient_vectors=5760,gradient_coordinates=64172160,physical_transitions=transition_count,optimizer_updates=0,goal_pass=False,test_opened=False);(out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'docs/technician_broker_removal_gradient_verification_v1_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS',mechanism_screen_pass=passed)))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
