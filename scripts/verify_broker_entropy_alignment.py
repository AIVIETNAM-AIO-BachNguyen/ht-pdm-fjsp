"""Independent manual MLP, RNG, closed-mass entropy gradients and primitive replay."""
import argparse,csv,hashlib,importlib.util,json,math,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];VERSION='technician_broker_entropy_alignment_diagnostic_v1';ARM='broker_marginal_ppo';TRAINS=tuple(range(310000000,310000003));STEPS=(0,15360,30720,61440,92160,122880,184320,245760);EARLY=STEPS[1:6]
def helpers():
    spec=importlib.util.spec_from_file_location('entropy_independent',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h
def near(a,b,label):
    if abs(float(a)-float(b))>1e-8:raise AssertionError(label)


def manual_logp(observations,weights,dtype,requires_grad=False):
    params={name:torch.tensor(value,dtype=dtype,requires_grad=requires_grad) for name,value in weights.items() if name.startswith('actor.')};state=torch.tensor(observations['state'],dtype=dtype);tech=torch.tensor(observations['technicians'],dtype=dtype);edges=torch.tensor(observations['edges'],dtype=dtype);legal=torch.tensor(observations['pair_mask'],dtype=torch.bool);result=[]
    for j in range(2):
        z=torch.cat([state,tech[:,j],edges[:,:,j].flatten(1),torch.zeros((len(state),12),dtype=dtype),torch.full((len(state),1),j/2,dtype=dtype)],-1)
        for layer in (0,2,4):
            z=torch.nn.functional.linear(z,params[f'actor.{layer}.weight'],params[f'actor.{layer}.bias'])
            if layer!=4:z=z.tanh()
        mask=torch.cat([torch.ones((len(state),1),dtype=torch.bool),legal[:,:,j]],-1);result.append(z.masked_fill(~mask,-1e9).log_softmax(-1))
    return torch.stack(result,1),params


def closed_masses(logp,observations,h):
    # For a fixed observed matching each idle agent may WAIT or lose to an
    # occupied machine. Closed preimage factors, no source matrix/entropy code.
    priority=np.array([[.5,.5],[.5,.25],[.5,.5],[.25,.5]]);legal=torch.tensor(observations['pair_mask'],dtype=torch.bool);terms=[]
    for pairs in h.baseline_catalogue():
        result=torch.zeros(len(logp),dtype=logp.dtype);winner={j:m for m,j in pairs}
        for j in range(2):
            if j in winner:result=result+logp[:,j,winner[j]+1]
            else:
                allowed=[0]+[m+1 for m,w in pairs if priority[m,w]>priority[m,j] or (priority[m,w]==priority[m,j] and w<j)]
                result=result+torch.logsumexp(logp[:,j,allowed],-1)
        terms.append(result)
    logs=torch.stack(terms,-1);return logs.exp(),logs


def reference_gradients(observations,actions,advantage,weights,h):
    logp,params=manual_logp(observations,weights,torch.float64,True);mass,logmass=closed_masses(logp,observations,h);p=logp.exp();hp=-(p*logp).sum(-1).mean(-1);hq=-(mass*logmass).sum(-1)/2;indices=[]
    for i,u in enumerate(actions):indices.append(h.baseline_catalogue().index(h.independent_decode(u,observations['pair_mask'][i],ARM,observations['edges'][i])[0]))
    actual_log=logmass[torch.arange(len(indices)),torch.tensor(indices)];a=torch.tensor(advantage.ravel(),dtype=torch.float64);vectors={};parameters=tuple(params.values())
    for name,value in (('reward',(actual_log*a).mean()),('proposal',hp.mean()),('physical',hq.mean())):vectors[name]=torch.cat([v.flatten() for v in torch.autograd.grad(value,parameters,retain_graph=True)]).detach().numpy()
    blocks=[];count=advantage.shape[1]
    for block in range(4):
        ids=(torch.arange(48)[:,None]*count+torch.arange(block*count//4,(block+1)*count//4)[None]).flatten();g=torch.autograd.grad((actual_log[ids]*a[ids]).mean(),parameters,retain_graph=block<3);blocks.append(torch.cat([v.flatten() for v in g]).detach().numpy())
    vectors['reward_blocks']=np.stack(blocks)
    return vectors,dict(hp=hp.detach().numpy(),hq=hq.detach().numpy(),matching_mass=mass.detach().numpy(),logq=actual_log.detach().numpy())


def metrics(vectors):
    r,p,q=[vectors[name].astype(float) for name in ('reward','proposal','physical')];rn,pn,qn=[float(np.dot(g,g)) for g in (r,p,q)];delta=q-p
    return dict(reward_squared_norm=rn,proposal_entropy_squared_norm=pn,physical_entropy_squared_norm=qn,entropy_gradients_nondegenerate=pn>1e-20 and qn>1e-20,entropy_gradient_cosine=float(np.dot(p,q)/math.sqrt(pn*qn)) if pn*qn>1e-40 else 0.,replacement_norm_ratio=.01*math.sqrt(float(np.dot(delta,delta)))/max(math.sqrt(rn),1e-20),projected_replacement=.01*float(np.dot(r,delta))/max(rn,1e-20))


def primary(records):
    means={};checks={}
    for t in TRAINS:
        rr=[r for r in records if r['train_seed']==t and r['checkpoint_steps'] in EARLY];assert len(rr)==5
        values=dict(relative_alias_gap=math.fsum(r['relative_alias_gap'] for r in rr)/5,opposed_entropy_checkpoints=sum(r['entropy_gradients_nondegenerate'] and r['entropy_gradient_cosine']<=0 for r in rr),material_replacement_checkpoints=sum(r['reward_squared_norm']>1e-20 and r['replacement_norm_ratio']>=.1 for r in rr),mean_projected_replacement=math.fsum(r['projected_replacement'] for r in rr)/5,all_reward_gradients_nondegenerate=all(r['reward_squared_norm']>1e-20 for r in rr));means[str(t)]=values;checks[str(t)]=dict(material_alias_gap=values['relative_alias_gap']>=.1,majority_entropy_opposition=values['opposed_entropy_checkpoints']>=3,majority_material_replacement=values['material_replacement_checkpoints']>=3,favorable_mean_projection=values['mean_projected_replacement']>=.01 and values['all_reward_gradients_nondegenerate'])
    return means,checks


def replay(directory,observations,arrays,episodes,t,step,weights,h,action_seed):
    policies=list(h.rows(directory/'policy.csv'));assert len(policies)==3072;policy={(int(r['episode']),int(r['tick'])):r for r in policies};assert len(policy)==3072;physics={};count=0
    for r in h.rows(directory/'physical.csv'):
        i,tick=int(r['episode']),int(r['tick']);s=json.loads(r['state']);pairs=tuple(map(tuple,json.loads(r['action'])));physical=h.seed(320100000+i,f'{h.VERSION}:N4:physical')
        if i not in physics:
            initial,events,pop=h.initial_events(physical);physics[i]=dict(state=initial,events=events,pop=pop,tick=0,metrics=defaultdict(float),rewards=np.zeros(48))
        ep=physics[i];assert s==ep['state'] and tick==ep['tick'] and int(r['physical_seed'])==physical and json.loads(r['events'])==ep['events'][tick] and r['phase']==('policy' if tick<48 else 'tail')
        if tick<48:
            pr=policy[i,tick];o=h.observation(s,tick);index=tick*64+i
            for name,value in o.items():assert np.array_equal(observations[name][index],value),name
            assert hashlib.sha256(o['state'].tobytes()).hexdigest()==pr['observation_sha256'];u=np.array(json.loads(pr['action']),int);assert np.array_equal(u,arrays['actions'][index]);decoded,_,diag=h.independent_decode(u,o['pair_mask'],ARM,o['edges']);assert decoded==pairs==tuple(map(tuple,json.loads(pr['pairs']))) and diag==json.loads(pr['diagnostics']);near(pr['log_probability'],arrays['old_log'][index],'stored behavior log');near(pr['value'],arrays['value'][tick,i],'stored value');near(pr['reward'],arrays['reward'][tick,i],'stored reward')
        else:
            score=np.array([[h.urgency(s,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);near(sum(score[m,j] for m,j in pairs),h.top_two_optimum(score,h.support(s)),'unchanged tail objective')
        following,parts=h.primitive_step(s,pairs,ep['events'][tick]);assert following==json.loads(r['following']);near(parts['objective'],r['cost'],'primitive cost')
        ep['rewards'][min(tick,47)]-=parts['objective']/20
        for k,v in parts.items():ep['metrics'][k]+=v
        ep['state']=following;ep['tick']+=1;count+=1
        if tick==47:ep['cutoff']=dict(ep['metrics']);ep['cutoff_state']=following
    assert count==3840 and set(physics)==set(range(64));lookup={int(r['episode']):r for r in episodes};assert len(lookup)==64
    for i,ep in physics.items():
        r=lookup[i];assert ep['tick']==60 and ep['pop']==r['population_sha256'];np.testing.assert_allclose(ep['rewards'],arrays['reward'][:,i],atol=1e-10,rtol=0)
        for k,v in ep['metrics'].items():near(v,r[k],'episode '+k)
        cut=ep['cutoff'];s=ep['cutoff_state'];pending=sum(s['failed'][m] and m not in s['assigned'] for m in range(4))
        for k,v in dict(availability=1-cut['unavailable_ticks']/192,terminal_pending=pending,policy_failures=cut['failures'],policy_cost=cut['objective'],tail_cost=ep['metrics']['objective']-cut['objective'],extended_cost=ep['metrics']['objective']).items():near(v,r[k],'cutoff '+k)
    # Exact source draw sequence reconstructed without importing its policy.
    rng=torch.Generator().manual_seed(action_seed)
    for tick in range(48):
        o={k:v[tick*64:(tick+1)*64] for k,v in observations.items()};logp,_=manual_logp(o,weights,torch.float32)
        expected=torch.stack([torch.multinomial(logp[:,j].exp(),1,generator=rng).squeeze(-1) for j in range(2)],-1).numpy();assert np.array_equal(expected,arrays['actions'][tick*64:(tick+1)*64]),'manual source stochastic RNG replay'
    expected_returns=np.stack([np.sum(arrays['reward'][tick:],axis=0) for tick in range(48)]);np.testing.assert_allclose(expected_returns,arrays['returns'],atol=1e-8,rtol=0)
    # Independent source32 recurrence / exact population normalization.
    r=torch.tensor(arrays['reward'],dtype=torch.float32);g=torch.zeros_like(r)
    for tick in range(47,-1,-1):g[tick]=r[tick]+(g[tick+1] if tick<47 else 0)
    a=(g-torch.tensor(arrays['value'],dtype=torch.float32)).flatten();a=(a-a.mean())/(a.std(unbiased=False)+1e-8);np.testing.assert_array_equal(a.numpy(),arrays['advantage'].ravel())
    return count


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);h=helpers();torch.set_num_threads(1);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False)
    def guard():assert time.monotonic()-start<900,'15minute verification hardstop'
    def bindings():
        assert h.sha(source/'manifest.json')==manifest['source_manifest_sha256']
        for name,meta in h.read(source/'manifest.json')['output_bindings'].items():guard();assert h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],name
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        m=h.read(source/'manifest.json');assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['updates']==0 and not m['test_opened'];bindings();parent=Path(m['source']);assert h.sha(parent/'manifest.json')==m['source_manifest_sha256'];inputs=h.read(source/'checkpoint_bindings.json');summary=h.read(source/'summary.json');episodes=list(h.rows(source/'episodes.csv'));assert len(inputs)==len(summary['records'])==24 and len(episodes)==1536;assert {(r['train_seed'],r['checkpoint_steps']) for r in inputs}=={(t,s) for t in TRAINS for s in STEPS};errors=defaultdict(float);records=[];transitions=0
        for item in tqdm(inputs,desc='24independent entropy/gradient/RNG replays'):
            guard();t=item['train_seed'];step=item['checkpoint_steps'];directory=source/str(t)/str(step);stats=h.read(directory/'statistics.json');w=h.weights(parent,item);obs=dict(np.load(directory/'observations.npz',allow_pickle=False));arrays=dict(np.load(directory/'arrays.npz',allow_pickle=False));stored=dict(np.load(directory/'gradients.npz',allow_pickle=False));action_seed=h.seed(320200000+TRAINS.index(t),f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:replica:{TRAINS.index(t)}:actions');assert action_seed==stats['action_seed'];transitions+=replay(directory,obs,arrays,[r for r in episodes if int(r['train_seed'])==t and int(r['checkpoint_steps'])==step],t,step,w,h,action_seed)
            expected_vectors,expected_arrays=reference_gradients(obs,arrays['actions'],arrays['advantage'],w,h)
            expected_value=h.mlp(obs['state'].astype(float),w,'critic').ravel()*115.2;assert np.all(np.abs(expected_value-arrays['value'].ravel())<=2e-5*115.2+1e-6*np.abs(arrays['value'].ravel())),'independent critic values';np.testing.assert_allclose(expected_arrays['logq'],arrays['old_log'],atol=2e-5,rtol=0)
            for name,value in expected_vectors.items():
                error=float(np.max(np.abs(value-stored[name])));errors['gradient_'+name]=max(errors['gradient_'+name],error);np.testing.assert_allclose(value,stored[name],atol=5e-6,rtol=2e-4)
            for name,value in expected_arrays.items():
                error=float(np.max(np.abs(value-arrays[name])));errors[name]=max(errors[name],error);np.testing.assert_allclose(value,arrays[name],atol=5e-6,rtol=0)
            probability=expected_arrays['matching_mass'];np.testing.assert_allclose(probability.sum(-1),1,atol=2e-12,rtol=0);assert np.all(expected_arrays['hp']>=expected_arrays['hq']-2e-12)
            raw=metrics(stored)
            for name,value in raw.items():
                if isinstance(value,bool):assert value==stats[name]
                else:near(value,stats[name],'gradient metric '+name)
            hp=torch.from_numpy(arrays['hp']).mean();hq=torch.from_numpy(arrays['hq']).mean();gap=torch.from_numpy(arrays['hp']-arrays['hq']).mean();record=dict(train_seed=t,checkpoint_steps=step,relative_alias_gap=float(hp-hq)/max(float(hp),1e-20),**raw);near(record['relative_alias_gap'],stats['relative_alias_gap'],'entropy relative gap');near(hp,stats['mean_proposal_entropy'],'Hp mean');near(hq,stats['mean_physical_entropy'],'Hq mean');near(gap,stats['mean_entropy_gap'],'entropy gap');near(np.abs(stored['reward_blocks'].mean(0)-stored['reward']).max(),stats['block_gradient_max_identity_error'],'block gradient identity');records.append(record)
            original=next(r for r in summary['records'] if r['train_seed']==t and r['checkpoint_steps']==step);assert stats==original
        means,checks=primary(records)
        for t,data in means.items():
            for k,value in data.items():near(value,summary['primary_replica_metrics'][t][k],'primary '+k)
        assert checks==summary['checks'] and all(all(v.values()) for v in checks.values())==summary['entropy_remedy_further_design_motivated'];bindings();result=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],models=24,episodes=1536,physical_transitions=transitions,policy_decisions=73728,actor_parameters=11141,maximum_errors=dict(errors),primary_replica_metrics=means,checks=checks,entropy_remedy_further_design_motivated=summary['entropy_remedy_further_design_motivated'],goal_pass=False,test_opened=False,limitation='Exact dense N4 diagnostic and finite-batch frozen actor directions, not causal learning improvement or scalable entropy method.')
        (out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py');shutil.copy2(ROOT/'docs/technician_broker_entropy_alignment_verification_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS')))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.output)
