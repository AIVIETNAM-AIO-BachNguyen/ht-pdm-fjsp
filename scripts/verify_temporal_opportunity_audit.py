"""Independent full16-event scalar Bellman replay of the fixed-state H4 audit."""
import argparse,csv,hashlib,itertools,json,math,shutil,subprocess,time
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import mean
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.technician_broker_marginal_env import scaled_config

ROOT=Path(__file__).resolve().parents[1];NAMES=('ages','failed','pending_wait','remaining','assigned');ORIGINS=('broker_marginal_ppo','ppo');TRAINS=tuple(range(290000000,290000003))
def check(ok,label):
    if not ok:raise AssertionError(label)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pack(s):return tuple(v for name in NAMES for v in s[name])
def unpack(k):return dict(ages=list(k[:4]),failed=list(k[4:8]),pending_wait=list(k[8:12]),remaining=list(k[12:14]),assigned=list(k[14:16]))
def pairs(value):return tuple(map(tuple,value))
def near(a,b,label):check(abs(float(a)-float(b))<1e-8,label)


class ScalarReference:
    def __init__(self,guard=lambda:None):self.cfg=scaled_config(4);self.guard=guard;self.expansions=0

    def actions(self,k):
        s=unpack(k);busy=set(s['assigned'])-{-1};result=[]
        for a,b in itertools.product(range(-1,4),repeat=2):
            if a==b and a>=0:continue
            p=tuple(sorted(([(a,0)] if a>=0 else [])+([(b,1)] if b>=0 else [])))
            if all(not s['remaining'][j] and m not in busy and (s['failed'][m] or s['ages'][m]>=3) for m,j in p):result.append(p)
        return sorted(result)

    @lru_cache(maxsize=None)
    def events(self,k,a):
        s=DispatchState(**{name:tuple(v) for name,v in unpack(k).items()});groups=defaultdict(lambda:[0.,0.])
        for e in itertools.product((False,True),repeat=4):
            following,cost,components=transition(self.cfg,s,a,e);near(cost,components['objective'],'scalar component cost');p=.45**sum(e)*.55**(4-sum(e));entry=groups[pack(asdict(following))];entry[0]+=p;entry[1]+=p*cost
        return tuple((following,p,weighted_cost) for following,(p,weighted_cost) in groups.items())

    def q(self,k,a,horizon):return math.fsum(weighted_cost+p*self.v(following,horizon-1) for following,p,weighted_cost in self.events(k,a))

    @lru_cache(maxsize=None)
    def v(self,k,horizon):
        if horizon==0:return 0.
        self.expansions+=1
        if self.expansions%500==0:self.guard()
        return min(self.q(k,a,horizon) for a in self.actions(k))


def independent_summary(rows):
    origins={}
    for origin in ORIGINS:
        rr=[r for r in rows if r['origin']==origin];means={key:mean(r[key] for r in rr) for key in ('joint_q','isolated_q','conditional_q','source_q')};per={}
        for t in TRAINS:
            group=[r for r in rr if r['train_seed']==t];iso=mean(r['isolated_q'] for r in group);conditional=mean(r['conditional_q'] for r in group);per[str(t)]=dict(roots=len(group),isolated_q=iso,conditional_q=conditional,gain=1-conditional/iso if iso else 0.)
        origins[origin]=dict(means=means,joint_gain_vs_isolated=1-means['joint_q']/means['isolated_q'],conditional_gain_vs_isolated=1-means['conditional_q']/means['isolated_q'],source_gap_to_joint=1-means['joint_q']/means['source_q'],conditional_better_roots=sum(r['isolated_q']-r['conditional_q']>1e-8 for r in rr),per_replica=per)
    g=origins[ORIGINS[0]];screen=dict(joint_gain_at_least_half_percent=g['joint_gain_vs_isolated']>=.005,conditional_gain_at_least_half_percent=g['conditional_gain_vs_isolated']>=.005,source_headroom_at_least_half_percent=g['source_gap_to_joint']>=.005,every_replica_conditional_gain_positive=all(r['gain']>0 for r in g['per_replica'].values()),at_least_six_conditional_improvements=g['conditional_better_roots']>=6)
    return origins,screen


def compare(a,b):
    if isinstance(a,dict):
        check(set(a)==set(b),'same summary fields')
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(int,float)) and not isinstance(a,bool):near(a,b,'summary arithmetic')
    else:check(a==b,'summary value')


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False,updates=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def guard():check(time.monotonic()-start<600,'10minute scalar verification hardstop')
    def bindings():
        check(sha(source/'manifest.json')==manifest['source_manifest_sha256'],'source manifest unchanged')
        for name,meta in read(source/'manifest.json')['output_bindings'].items():guard();check(sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'source output binding '+name)
    try:
        m=read(source/'manifest.json');check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened'] and m['updates']==0,'completed nonlearning source');bindings()
        parent=Path(m['source']);check(sha(parent/'manifest.json')==m['source_manifest_sha256'],'frozen parent binding');contexts=read(source/'contexts.json');rows=read(source/'roots.json');expected={(a,t,s,tick) for a in ORIGINS for t in TRAINS for s in range(300100000,300100004) for tick in (0,12,24,36)}
        def label(r):return r['origin'],r['train_seed'],r['seed'],r['tick']
        context={label(r):r for r in contexts};result={label(r):r for r in rows};check(len(contexts)==96 and len(rows)==96 and set(context)==set(result)==expected,'fixed96source grid')
        with (parent/'physical.csv').open() as f:
            found=set()
            for r in csv.DictReader(f):
                k=(r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']))
                if r['mode']=='greedy' and k in expected:
                    check(context[k]['state']==json.loads(r['state']) and context[k]['source_pairs']==json.loads(r['pairs']),'original physical context');found.add(k)
            check(found==expected,'all contexts traced')
        engine=ScalarReference(guard);errors=[]
        for r in tqdm(rows,desc='Independent scalar H4 /96 roots',mininterval=1):
            guard();root=context[label(r)];s=root['state'];k=pack(s);check(hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()==r['state_sha256'],'root state digest');actions=engine.actions(k);qs={a:engine.q(k,a,4) for a in actions}
            for name in ('joint','isolated','conditional','source'):
                a=pairs(r[name+'_pairs']);check(a in qs,'root action support');error=abs(qs[a]-r[name+'_q']);check(error<1e-8,'independent H4 '+name);errors.append(error)
            near(min(qs.values()),r['joint_q'],'global H4 minimum');check(len(actions)==r['legal_matchings'],'matching catalogue support')
            for j,token in enumerate(r['isolated_tokens']):
                single=[()] + [((mm,j),) for mm in range(4) if ((mm,j),) in qs];chosen=((token-1,j),) if token else ();near(qs[chosen],min(qs[a] for a in single),'isolated Q argmin')
            tokens=r['isolated_tokens'];proposals=[(token-1,j) for j,token in enumerate(tokens) if token];decoded=[];D=[[1,2],[1,4],[2,1],[4,1]];R=[[.5,1],[.5,1],[1,.5],[1,.5]]
            for mm in set(mm for mm,j in proposals):decoded.append(max((a for a in proposals if a[0]==mm),key=lambda a:(R[a[0]][a[1]]/D[a[0]][a[1]],-a[1])))
            check(tuple(sorted(decoded))==pairs(r['isolated_pairs']),'quality broker')
            prefix=((tokens[0]-1,0),) if tokens[0] else ();conditional=pairs(r['conditional_pairs']);check(r['conditional_tokens'][0]==tokens[0] and tuple(a for a in conditional if a[1]==0)==prefix,'first commitment fixed');allowed=[a for a in actions if tuple(p for p in a if p[1]==0)==prefix];near(qs[conditional],min(qs[a] for a in allowed),'conditional Q argmin')
        origins,screen=independent_summary(rows);original=read(source/'summary.json');compare(origins,original['origins']);compare(screen,original['screen']);check(all(screen.values())==original['conditional_q_mechanism_further_design_motivated'],'locked verdict');bindings()
        verification=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],roots=96,root_q_values=384,max_absolute_q_error=max(errors),scalar_bellman_expansions=engine.expansions,scalar_transition_groups=engine.events.cache_info().currsize,origins=origins,screen=screen,goal_pass=False,test_opened=False,limitation='Independent exact H4 calculation, not H60 policy improvement, health confirmation or novel algorithm evidence.')
        (out/'verification.json').write_text(json.dumps(verification,indent=2));shutil.copy2(__file__,out/'verifier.py');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS')))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.output)
