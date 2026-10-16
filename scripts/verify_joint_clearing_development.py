"""Independent primitive dynamics, NumPy policy and artifact verification."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,shutil,subprocess,time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

VERSION='technician_joint_clearing_ppo_v1'
COMMIT='690be2d8241b9e5a2625c537e0c684f95674b4d0'
CORE=('joint_clearing_ppo','forward_clearing_ppo','reverse_clearing_ppo','independent_clearing_ppo')
ARMS=(*CORE,'ppo','normalized_ppo','mappo','full_mappo')
FROZEN={f'frozen_{a}':a for a in CORE[1:]}
TRAINS=tuple(range(250000000,250000003));DEV=tuple(range(250100000,250100032));STEPS=(0,122880,245760,368640,491520)
D=np.array([[1,2],[1,4],[2,1],[4,1]],int)
R=np.array([[.5,1.],[.5,1.],[1.,.5],[1.,.5]])
ROOT=Path(__file__).resolve().parents[1]

def check(ok,message):
    if not ok:raise AssertionError(message)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(p):
    with p.open() as f:yield from csv.DictReader(f)
def near(a,b,label):check(abs(float(a)-float(b))<1e-8,label)
def seed(raw,role):return int.from_bytes(hashlib.sha256(f'{raw}:{role}'.encode()).digest()[:8],'big')


def initial_events(physical):
    rng=random.Random(seed(physical,'initial'));events_rng=random.Random(seed(physical,'failures'))
    state=dict(ages=[rng.randrange(4) for _ in range(4)],failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2)
    events=[[events_rng.random()<.45 for _ in range(4)] for _ in range(60)]
    population=hashlib.sha256(json.dumps(dict(seed=physical,state=state,events=events),sort_keys=True).encode()).hexdigest()
    return state,events,population


def support(state,urgent=True):
    busy={m for m in state['assigned'] if m>=0}
    return np.array([[m not in busy and state['remaining'][j]==0 and (not urgent or state['failed'][m] or state['ages'][m]>=3)
                     for j in range(2)] for m in range(4)],bool)


def primitive_step(state,pairs,events):
    """Pure dictionaries, no repository simulator imports."""
    check(len({m for m,j in pairs})==len(pairs) and len({j for m,j in pairs})==len(pairs),'matching uniqueness')
    legal=support(state,False);check(all(legal[m,j] for m,j in pairs),'physical eligibility')
    following={k:list(v) for k,v in state.items()};pre_failed=list(state['failed'])
    pm=sum(not pre_failed[m] for m,j in pairs);cm=len(pairs)-pm
    for m,j in pairs:following['assigned'][j]=m;following['remaining'][j]=int(D[m,j])
    serving={m for m in following['assigned'] if m>=0}
    unavailable=sum(pre_failed[m] or m in serving for m in range(4));waiting=sum(pre_failed[m] and m not in serving for m in range(4))
    overdue=sum(pre_failed[m] and m not in serving and state['pending_wait'][m]+1>4 for m in range(4));failures=0;complete=0
    for m in range(4):
        if m in serving:continue
        if pre_failed[m]:following['pending_wait'][m]+=1
        else:
            following['ages'][m]=min(8,state['ages'][m]+1)
            if following['ages'][m]>=4 and events[m]:following['failed'][m]=True;following['pending_wait'][m]=0;failures+=1
    for j,m in enumerate(following['assigned']):
        if m<0:continue
        following['remaining'][j]-=1
        if following['remaining'][j]==0:
            following['ages'][m]=math.floor(following['ages'][m]*(1-R[m,j]));following['failed'][m]=False
            following['pending_wait'][m]=0;following['assigned'][j]=-1;complete+=1
    base=pm+2*cm+12*unavailable+15*failures
    metrics=dict(objective=base+12*overdue,maintenance_cost=pm+2*cm,unavailability_cost=12*unavailable,
        failure_cost=15*failures,preventive_jobs=pm,corrective_jobs=cm,jobs=len(pairs),completed_jobs=complete,
        failures=failures,unavailable_ticks=unavailable,failed_waiting_ticks=waiting,service_ticks=len(serving),
        invalid_assignments=0,base_cost=base,waiting_cost=12*overdue,overdue_waiting_ticks=overdue,service_adjusted_cost=base+12*overdue)
    return following,metrics


def observation(state,tick):
    owners={m:j for j,m in enumerate(state['assigned']) if m>=0}
    machines=np.asarray([[state['ages'][m]/8,float(state['failed'][m]),state['pending_wait'][m]/12,
        float(m in owners),state['remaining'][owners[m]]/12 if m in owners else 0,
        .45 if not state['failed'][m] and m not in owners and state['ages'][m]+1>=4 else 0] for m in range(4)],np.float32)
    technicians=np.asarray([[float(state['remaining'][j]==0),state['remaining'][j]/12,
        float(state['failed'][m]) if (m:=state['assigned'][j])>=0 else 0] for j in range(2)],np.float32)
    edges=np.stack([np.ones((4,2)),D/12,R,np.array([[state['assigned'][j]==m for j in range(2)] for m in range(4)])],-1).astype(np.float32)
    global_features=np.asarray([(48-tick)/12,.5,.45,.05,.1,.6,.75,4/12,.6],np.float32)
    vector=np.concatenate([machines.ravel(),technicians.ravel(),edges.ravel(),global_features])
    return dict(machines=machines,technicians=technicians,edges=edges,global_features=global_features,state=vector,pair_mask=support(state))


def best(q,mask,j,excluded=()):
    choices=[m for m in range(4) if mask[m,j] and q[m,j]>0 and m not in excluded]
    return max(choices,key=lambda m:(q[m,j],-m)) if choices else None


def top_two_optimum(q,mask):
    """Exact K2 objective bound using alternatives, no matching enumeration."""
    picks=[best(q,mask,j) for j in range(2)];values=[float(q[m,j]) if m is not None else 0 for j,m in enumerate(picks)]
    if picks[0] is None or picks[1] is None or picks[0]!=picks[1]:return sum(values)
    m=picks[0];second=[best(q,mask,j,(m,)) for j in range(2)]
    rest=[float(q[a,j]) if a is not None else 0 for j,a in enumerate(second)]
    return max(values[0]+rest[1],rest[0]+values[1])


def verify_decoder(utility,mask,arm,pairs,diag):
    u=np.asarray(utility,float);check(u.shape==(4,2) and np.isfinite(u).all() and np.abs(u).max()<=1e6 and (u[~mask]==0).all(),'latent support')
    q=np.rint(u*1e6).astype(np.int64);actual=sum(int(q[m,j]) for m,j in pairs)
    check(all(mask[m,j] and q[m,j]>0 for m,j in pairs),'positive legal cleared bids')
    proposals=[(m,j) for j in range(2) if (m:=best(q,mask,j)) is not None]
    excess=len(proposals)-len({m for m,j in proposals})
    if arm==CORE[0]:check(actual==top_two_optimum(q,mask),'global utility optimum')
    elif arm==CORE[3]:
        expected=[]
        for m in range(4):
            group=[p for p in proposals if p[0]==m]
            if group:expected.append(max(group,key=lambda p:(q[p],-p[1])))
        check(tuple(sorted(expected))==pairs,'independent arbitration')
    else:
        expected=[]
        for j in (range(2) if arm==CORE[1] else (1,0)):
            m=best(q,mask,j,{m for m,j in expected})
            if m is not None:expected.append((m,j))
        check(tuple(sorted(expected))==pairs,'serial clearing')
    expected_diag=dict(conflicts=int(excess>0) if arm==CORE[3] else 0,rejected_proposals=excess if arm==CORE[3] else 0,
        uncoordinated_proposal_rejections=excess,executed_starts=len(pairs),quantized_utility=actual,
        order_forward=int(arm==CORE[1]),order_reverse=int(arm==CORE[2]))
    check(diag==expected_diag,'decoder diagnostics')


def mlp(array,weights,prefix):
    for i in (0,2,4):
        array=array@weights[f'{prefix}.{i}.weight'].T+weights[f'{prefix}.{i}.bias']
        if i!=4:array=np.tanh(array)
    return array


def utility_mean(observations,weights):
    machines=np.stack([o['machines'] for o in observations]).astype(float);tech=np.stack([o['technicians'] for o in observations]).astype(float)
    edges=np.stack([o['edges'] for o in observations]).astype(float);g=np.stack([o['global_features'] for o in observations]).astype(float);b=len(observations)
    raw=np.concatenate([np.broadcast_to(machines[:,:,None],(b,4,2,6)),np.broadcast_to(tech[:,None],(b,4,2,3)),edges,np.broadcast_to(g[:,None,None],(b,4,2,9))],-1)
    edge=mlp(raw,weights,'encoder')
    context=np.concatenate([edge,np.broadcast_to(edge.mean(2)[:,:,None],edge.shape),np.broadcast_to(edge.mean(1)[:,None],edge.shape),np.broadcast_to(edge.mean((1,2))[:,None,None],edge.shape)],-1)
    return np.where(np.stack([o['pair_mask'] for o in observations]),mlp(context,weights,'bid')[...,0],0.)


def independent_initial(arm,train):
    import torch
    from torch import nn
    def net(i,h,o):return nn.Sequential(nn.Linear(i,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,o))
    role='graph' if arm in CORE else 'ppo' if arm=='normalized_ppo' else arm
    initialization=seed(train,f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:initialization:{role}')
    with torch.random.fork_rng():
        torch.manual_seed(initialization);m=nn.Module()
        if arm in CORE:
            m.encoder=net(22,32,32);m.bid=net(128,32,1);m.log_scale=nn.Parameter(torch.zeros(()))
            m.value_encoder=net(22,32,32);m.value_calendar_encoder=net(4,32,32);m.value_context=net(65,32,32);m.value_head=net(41,32,1)
        else:
            m.actor=net(71 if arm in ('ppo','normalized_ppo') else 33,64,21 if arm in ('ppo','normalized_ppo') else 3);m.critic=net(71,64,1)
            if arm=='full_mappo':m.actor=net(75,64,3)
            if arm!='ppo':
                critic=m.critic;wrapped=nn.Module();wrapped.net=critic;m.critic=wrapped
    return m.state_dict()


def weights(source,choice):
    import torch
    payload=torch.load(source/choice['path'],map_location='cpu',weights_only=True);h=hashlib.sha256()
    for name,v in payload['state_dict'].items():h.update(name.encode());h.update(v.numpy().tobytes())
    check(h.hexdigest()==payload['parameter_sha256']==choice['parameter_sha256'] and sha(source/choice['path'])==choice['sha256'],'checkpoint binding/digest')
    return {k:v.numpy().astype(float) for k,v in payload['state_dict'].items()}


def verify_actor(observations,actions,arm,w):
    if arm in CORE:
        expected=utility_mean(observations,w);actual=np.asarray(actions,float);error=np.abs(expected-actual)
        ratio=error/(5e-6+1e-6*np.abs(actual));check(ratio.max()<=1,'NumPy utility forward')
        return float(error.max())
    if arm in ('ppo','normalized_ppo'):
        logits=mlp(np.stack([o['state'] for o in observations]).astype(float),w,'actor');catalogue=baseline_catalogue()
        mask=np.array([[all(o['pair_mask'][m,j] for m,j in a) for a in catalogue] for o in observations])
        selected=np.asarray(actions,int)
        check(mask[np.arange(len(actions)),selected].all(),'PPO token support');chosen=logits[np.arange(len(actions)),selected]
        deficit=np.where(mask,logits,-1e9).max(1)-chosen
    else:
        inputs=[]
        for o in observations:
            if arm=='full_mappo':local=np.concatenate([np.tile(o['state'],(4,1)),np.eye(4,dtype=np.float32)],-1)
            else:
                workers=o['technicians'].copy();workers[:,2]=0
                local=np.concatenate([o['machines'],np.tile(workers.ravel(),(4,1)),o['edges'].reshape(4,-1),np.tile(o['global_features'],(4,1)),np.eye(4,dtype=np.float32)],-1)
            inputs.append(local)
        logits=mlp(np.stack(inputs).astype(float),w,'actor');mask=np.stack([np.c_[np.ones(4,bool),o['pair_mask']] for o in observations]);selected=np.asarray(actions,int)
        chosen=np.take_along_axis(logits,selected[...,None],-1)[...,0]
        check(np.take_along_axis(mask,selected[...,None],-1).all(),'MAPPO token support');deficit=np.where(mask,logits,-1e9).max(-1)-chosen
    ratio=deficit/(5e-6+1e-6*np.abs(chosen));check(ratio.max()<=1,'NumPy baseline greedy logits')
    return float(deficit.max())


@lru_cache(maxsize=1)
def baseline_catalogue():
    """Only the existing original PPO baseline's21-token interface."""
    result=[]
    for a in range(-1,4):
        for b in range(-1,4):
            if a>=0 and a==b:continue
            result.append(tuple(sorted([(a,0)] if a>=0 else [] )+sorted([(b,1)] if b>=0 else [])))
    return tuple(sorted(tuple(sorted(p)) for p in result))


def urgency(state,m):return 100*state['failed'][m]+4.5*(not state['failed'][m] and state['ages'][m]+1>=4)+state['ages'][m]/8+20*max(0,state['pending_wait'][m]-3)


def verify_baseline(action,mask,arm,state,pairs):
    if arm in ('ppo','normalized_ppo'):expected=baseline_catalogue()[int(action)]
    elif arm in ('mappo','full_mappo'):
        requests=[(m,int(j)-1) for m,j in enumerate(action) if j];expected=[]
        check(all(mask[m,j] for m,j in requests),'MAPPO request support')
        for j in range(2):
            group=[p for p in requests if p[1]==j]
            if group:expected.append(max(group,key=lambda p:(urgency(state,p[0]),-p[0])))
        expected=tuple(sorted(expected))
    else:raise AssertionError('unknown baseline')
    check(expected==pairs,'baseline decoder')


def endpoint_gates(means):
    p=means[CORE[0]];points={};checks={}
    for arm in ARMS[1:]:
        v=1-p['normalized_cost']/means[arm]['normalized_cost'];key='gain_vs_'+arm;points[key]=v
        checks[key]=v>=.005 if arm==CORE[3] else v>0 if arm in ('mappo','full_mappo') else v>=-.01
    for arm in ('ppo','normalized_ppo'):
        for metric in ('availability','normalized_pending','normalized_failures'):
            key=metric+'_delta_vs_'+arm;v=p[metric]-means[arm][metric];points[key]=v
            checks[key]=v>=-.01 if metric=='availability' else v<=.025 if metric=='normalized_pending' else v<=.25
    check(len(checks)==13,'13endpoints');return points,checks


def check_bindings(source,guard):
    m=read(source/'manifest.json');check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened'] and m['git']['commit']==COMMIT and not m['git']['dirty'],'terminal frozen source')
    for name,meta in tqdm(m['output_bindings'].items(),desc='All source bindings'):
        guard();p=source/name;check(sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes'],'output binding '+name)
    for name,h in m['source_seal'].items():
        check(sha(source/name)==h,'source seal')
        if name.startswith('source_snapshot/'):
            blob=subprocess.check_output(['git','show',f'{COMMIT}:src/ht_pdm_fjsp/{Path(name).name}'],cwd=ROOT)
            check(hashlib.sha256(blob).hexdigest()==h,'committed source blob')
            check(sha(ROOT/'src/ht_pdm_fjsp'/Path(name).name)==h,'local frozen timing source')
    for name,relative in (('protocol.md',f'docs/{VERSION}_plan.md'),('seed_registry.json',f'configs/{VERSION}_seed_registry.json')):
        blob=subprocess.check_output(['git','show',f'{COMMIT}:{relative}'],cwd=ROOT);check(hashlib.sha256(blob).hexdigest()==sha(source/name),'committed protocol/registry')
    return m


def verify_budgets(source,c,choices,guard):
    import torch
    torch.set_num_threads(1)
    for key,value in dict(env_steps=491520,vector_envs=16,rollout=3072,epochs=4,minibatch=256,hidden=32,baseline_hidden=64,gamma=1.,learning_rate=.0003,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,teacher_assistance=False,pretraining_steps=0,training_processes=8).items():check(c[key]==value,'locked config '+key)
    check(c['train_seeds']==list(TRAINS) and c['development_seeds']==list(DEV) and c['test_seeds']==[] and c['checkpoints']==list(STEPS[1:]),'seed/checkpoint panel')
    audits=read(source/'training_audit.json');check(len(audits)==24 and len(choices)==24,'fit grid')
    check({(a['algorithm'],a['train_seed']) for a in audits}=={(b,t) for b in ARMS for t in TRAINS},'unique fit audit grid')
    physics=read(source/'physics.json')
    check(physics==dict(machines=4,technicians=2,horizon=60,failure_age=4,failure_probability=.45,service_time=D.tolist(),restoration=R.tolist(),
        max_age=8,preventive_cost=1.,corrective_cost=2.,unavailable_cost=12.,failure_cost=15.,waiting_limit=4,waiting_price=12.),'frozen physical configuration')
    for a in audits:
        check((a['algorithm'],a['train_seed']) in {(b,t) for b in ARMS for t in TRAINS} and a['from_scratch'] and a['pretraining_steps']==0 and a['learned_steps']==491520 and a['physical_ticks']==614400 and a['training_episodes']==10240 and a['optimizer_steps']==7680 and a['changed'] and a['behavior_normalized_max_error']<=1,'fit budget')
    counts=read(source/'parameter_counts.json');check(all(counts[a]==19843 for a in CORE),'core equal parameter count')
    for t in tqdm(TRAINS,desc='Independent initial tensors'):
        for a in ARMS:
            expected=independent_initial(a,t);p=source/'N4'/a/str(t)/'checkpoints/step_000000000.pt';data=torch.load(p,map_location='cpu',weights_only=True)
            check(list(expected)==list(data['state_dict']) and all(torch.equal(expected[k],data['state_dict'][k]) for k in expected),'fresh random tensors '+a)
            final=torch.load(source/'N4'/a/str(t)/'checkpoints/step_000491520.pt',map_location='cpu',weights_only=True);h=hashlib.sha256()
            for name,v in final['state_dict'].items():h.update(name.encode());h.update(v.numpy().tobytes())
            audit=[v for v in audits if v['algorithm']==a and v['train_seed']==t][0]
            check(h.hexdigest()==audit['parameter_sha256']==final['parameter_sha256'] and final['steps']==491520 and final['train_seed']==t and final['algorithm']==a,'final update/checkpoint digest')
    progress=defaultdict(list)
    for r in rows(source/'training_progress.csv'):progress[r['algorithm'],int(r['train_seed'])].append(r)
    for a in ARMS:
        for t in TRAINS:
            group=progress[a,t];check([int(r['learned_steps']) for r in group]==list(range(3072,491521,3072)),'160complete updates')
            check(all(int(r['optimizer_steps'])==int(r['learned_steps'])//3072*48 and float(r['behavior_normalized_max_error'])<=1 for r in group),'update replay/budget')
            if a in CORE:check(all(sum(float(r[k]) for r in group)>0 for k in ('actor_update_norm','critic_update_norm')),'genuine actor/critic updates')
    seen=set();paired={};populations={}
    for r in tqdm(rows(source/'training_episodes.csv'),total=245760,desc='Paired training tapes'):
        if len(seen)%5000==0:guard()
        a,t,w,e=r['algorithm'],int(r['train_seed']),int(r['worker']),int(r['episode_index']);key=t,w,e
        check((a,*key) not in seen and a in ARMS and t in TRAINS and 0<=w<16 and 0<=e<640,'train episode grid');seen.add((a,*key))
        physical=seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}');check(physical==int(r['physical_seed']),'training seed namespace')
        if key not in populations:populations[key]=initial_events(physical)[2]
        check(r['population_sha256']==populations[key] and paired.setdefault(key,r['population_sha256'])==r['population_sha256'],'training tape reconstruction/pairing')
        check(float(r['invalid_assignments'])==0 and int(r['physical_ticks'])==60 and abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5,'train reward/validity')
    check(len(seen)==245760 and len(paired)==30720,'complete training grid')
    dev=list(rows(source/'development_episodes.csv'));check(len(dev)==3840,'DEV count');grid=set()
    for r in dev:
        key=r['method'],int(r['train_seed']),int(r['checkpoint_steps']),int(r['seed']);check(key not in grid,'duplicate DEV');grid.add(key)
    check(grid=={(a,t,step,s) for a in ARMS for t in TRAINS for step in STEPS for s in DEV},'full DEV grid')
    for a in ARMS:
        for t in TRAINS:
            score={step:np.mean([float(r['extended_cost']) for r in dev if r['method']==a and int(r['train_seed'])==t and int(r['checkpoint_steps'])==step]) for step in STEPS}
            best_step=min(score,key=lambda step:(score[step],step));choice=choices[f'{a}:{t}']
            check(choice['steps']==best_step and choice['development_cost']==score[best_step],'original checkpoint choice');weights(source,choice)
    return dict(models=24,training_episodes=len(seen),training_decisions=11796480,adam_steps=184320,paired_training_tapes=len(paired),development_episodes=len(dev))


def verify_selected(source,choices,guard):
    selected=list(rows(source/'selected_episodes.csv'));lookup={(r['method'],int(r['train_seed']),int(r['seed'])):r for r in selected}
    expected={(a,t,s) for a in (*ARMS,*FROZEN) for t in TRAINS for s in DEV}|{('deadline',-1,s) for s in DEV}
    check(len(selected)==1088 and set(lookup)==expected,'selected complete grid')
    coordination={}
    for r in rows(source/'coordination.csv'):
        key=r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']);check(key not in coordination,'duplicate coordination');coordination[key]=r
    episodes={};actor_data=defaultdict(list);ticks=0;core_decisions=0
    for r in tqdm(rows(source/'decisions.csv'),total=65280,desc='Primitive physical replay'):
        if ticks%1000==0:guard()
        key=r['method'],int(r['train_seed']),int(r['seed']);tick=int(r['tick']);state=json.loads(r['state']);pairs=tuple(map(tuple,json.loads(r['action'])))
        check(key in lookup and key[2] in DEV,'DEV only physical trace')
        if key not in episodes:
            physical=seed(key[2],f'{VERSION}:N4:physical');initial,events,population=initial_events(physical)
            episodes[key]=dict(state=initial,events=events,tick=0,population=population,physical=physical,metrics=defaultdict(float),broker=defaultdict(int))
        ep=episodes[key];check(state==ep['state'] and tick==ep['tick'] and json.loads(r['events'])==ep['events'][tick],'independent state/event chain')
        check(int(r['physical_seed'])==ep['physical'],'physical trace seed')
        check(all(support(state)[m,j] for m,j in pairs),'urgent matching support')
        if tick<48:
            cr=coordination[key+(tick,)];obs=observation(state,tick);check(hashlib.sha256(obs['state'].tobytes()).hexdigest()==cr['observation_sha256'],'independent observation')
            check(tuple(map(tuple,json.loads(cr['pairs'])))==pairs,'coordination/physical matching')
            action=json.loads(cr['submitted']);diag=json.loads(cr['diagnostics']);arm=FROZEN.get(key[0],key[0])
            if arm in CORE:verify_decoder(action,obs['pair_mask'],arm,pairs,diag);core_decisions+=1
            elif arm!='deadline':verify_baseline(action,obs['pair_mask'],arm,state,pairs)
            else:
                score=np.array([[urgency(state,m)+R[m,j]/D[m,j] for j in range(2)] for m in range(4)])
                near(sum(score[m,j] for m,j in pairs),top_two_optimum(score,obs['pair_mask']),'deadline objective')
            for field in ('order_forward','order_reverse','conflicts','rejected_proposals'):ep['broker'][field]+=diag[field]
            if arm!='deadline':actor_data[key[0],key[1]].append((obs,action))
        else:
            score=np.array([[urgency(state,m)+R[m,j]/D[m,j] for j in range(2)] for m in range(4)])
            near(sum(score[m,j] for m,j in pairs),top_two_optimum(score,support(state)),'tail deadline objective')
        following,metrics=primitive_step(state,pairs,ep['events'][tick]);check(following==json.loads(r['following']),'independent following state');near(metrics['objective'],r['cost'],'independent transition cost')
        for k,v in metrics.items():ep['metrics'][k]+=v
        ep['state']=following;ep['tick']+=1;ticks+=1
        if tick==47:ep['cutoff']=dict(ep['metrics']);ep['cutoff_state']=following
    check(ticks==65280 and len(coordination)==52224 and len(episodes)==1088,'complete replay grid')
    for key,ep in episodes.items():
        r=lookup[key];check(ep['tick']==60 and ep['population']==r['population_sha256'],'population/final tick')
        for metric,v in ep['metrics'].items():near(v,r[metric],'aggregate '+metric)
        cutoff=ep['cutoff'];state=ep['cutoff_state'];pending=sum(state['failed'][m] and m not in state['assigned'] for m in range(4))
        for metric,v in dict(policy_cost=cutoff['objective'],tail_cost=ep['metrics']['objective']-cutoff['objective'],extended_cost=ep['metrics']['objective'],
                availability=1-cutoff['unavailable_ticks']/192,policy_failures=cutoff['failures'],terminal_pending=pending,
                normalized_cost=ep['metrics']['objective']/4,normalized_pending=pending/4,normalized_failures=cutoff['failures']/4).items():near(v,r[metric],'cutoff health '+metric)
        for field,v in ep['broker'].items():near(v,r['broker_'+field],'broker total')
    errors={}
    for (label,t),data in tqdm(actor_data.items(),desc='NumPy actor forward'):
        guard();arm=CORE[0] if label in FROZEN else label;w=weights(source,choices[f'{arm}:{t}'])
        errors[f'{label}:{t}']=verify_actor([o for o,a in data],[a for o,a in data],arm,w)
    means={a:{k:float(np.mean([float(r[k]) for r in selected if r['method']==a])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in (*ARMS,*FROZEN,'deadline')}
    per_replica={a:{str(t):{k:float(np.mean([float(r[k]) for r in selected if r['method']==a and int(r['train_seed'])==t])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for t in TRAINS} for a in (*ARMS,*FROZEN)}
    return dict(episodes=1088,physical_transitions=ticks,coordination_decisions=len(coordination),core_decisions=core_decisions,numpy_forward_errors=errors,means=means,per_replica=per_replica)


def independent_timing(source,choices,guard):
    import torch
    from ht_pdm_fjsp.technician_joint_clearing_policy import JointClearingPolicy,decode_utility
    process_lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    check(not any('from multiprocessing.spawn import spawn_main' in line for line in process_lines),'training workers still active during timing')
    torch.set_num_threads(1);samples=[]
    for t in TRAINS:
        choice=choices[f'{CORE[0]}:{t}'];payload=torch.load(source/choice['path'],map_location='cpu',weights_only=True);m=JointClearingPolicy(CORE[0]);m.load_state_dict(payload['state_dict']);m.eval();meta=choice['observations']
        check(meta['states']==48 and meta['seed']==DEV[0] and sha(source/meta['path'])==meta['sha256'],'timing observations binding')
        with np.load(source/meta['path'],allow_pickle=False) as f:obs=[{k:f[k][i].copy() for k in f.files} for i in range(48)]
        for o in obs[:8]:decode_utility(m.select(o),o['pair_mask'],CORE[0])
        for repeat in range(3):
            for i,o in enumerate(obs):
                guard();start=time.perf_counter_ns();decode_utility(m.select(o),o['pair_mask'],CORE[0]);elapsed=(time.perf_counter_ns()-start)/1e6
                samples.append(dict(train_seed=t,repeat=repeat,state_index=i,milliseconds=elapsed))
    values=[r['milliseconds'] for r in samples]
    return dict(samples=432,p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)),passes_50ms=bool(np.percentile(values,95)<=50)),samples


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=sha(source/'manifest.json'))
    def guard():check(time.monotonic()-start<900,'15min verification stop')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        m=check_bindings(source,guard);c=read(source/'resolved_config.json');choices=read(source/'checkpoint_choices.json')
        budgets=verify_budgets(source,c,choices,guard);selected=verify_selected(source,choices,guard)
        summary=read(source/'summary.json');points,checks=endpoint_gates(selected['means'])
        for key,v in points.items():near(v,summary['endpoints'][key],'point '+key)
        check(checks==summary['checks'],'13endpoint decisions')
        for a,values in selected['means'].items():
            for key,v in values.items():near(v,summary['means'][a][key],'method means')
        latency=read(source/'latency.json');check(latency['samples']==432 and summary['latency_pass']==(latency['p95_ms']<=50),'primary timing gate')
        passed=all(checks.values()) and latency['p95_ms']<=50
        check(summary['development_screen_pass']==passed and m['development_screen_pass']==passed and not summary['goal_pass'] and not summary['test_opened'],'original DEV verdict')
        timing,samples=independent_timing(source,choices,guard)
        with (out/'timing_samples.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(samples[0]));writer.writeheader();writer.writerows(samples)
        result=dict(status='PASS',classification='DEVELOPMENT_VERIFICATION_ONLY',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],
            source_bindings_verified=len(m['output_bindings']),budgets=budgets,selected=selected,independent_points=points,independent_checks=checks,
            original_development_pass=passed,primary_latency=latency,independent_latency=timing,goal_pass=False,test_opened=False,
            limitations=['No population CI from DEV.','Historical training actions/minibatch gradients not retained; recorded replay audits and correctness tests cover that path.',
                'Independent utility-optimality check proves learned-utility objective, not future maintenance optimality.','Timing remeasurement is separate and never rewrites original verdict.'])
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,out/'verifier.py')
        shutil.copy2(ROOT/'docs/technician_joint_clearing_v1_verification_plan.md',out/'verification_plan.md')
        check(sha(source/'manifest.json')==manifest['source_manifest_sha256'],'source unchanged')
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(output=str(out),status='PASS',original_development_pass=passed,checks=checks)));return result
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)

if __name__=='__main__':main()
