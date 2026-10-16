"""Fixed-state exact H4 allocation diagnostic; no learning or scalable-policy claim."""
import argparse,csv,hashlib,importlib.util,itertools,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2];VERSION='technician_temporal_opportunity_audit_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';ORIGINS=('broker_marginal_ppo','ppo')
TRAINS=tuple(range(290000000,290000003));SEEDS=tuple(range(300100000,300100004));TICKS=(0,12,24,36)
SOURCE_COMMIT='539131365a5f42ca33e0b439d6a952ca4d7c10f5'
FIELDS=('ages','failed','pending_wait','remaining','assigned')


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False))
def helpers():
    spec=importlib.util.spec_from_file_location('temporal_primitives',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h
def key(state):return tuple(int(v) for field in FIELDS for v in state[field])
def state(k):return dict(ages=list(k[:4]),failed=[bool(v) for v in k[4:8]],pending_wait=list(k[8:12]),remaining=list(k[12:14]),assigned=list(k[14:16]))


class Exact:
    def __init__(self,h,guard=lambda:None,limit=250000):self.h=h;self.guard=guard;self.limit=limit;self.cache={};self.expansions=0;self.hits=0;self.branches=0

    def actions(self,s):
        mask=self.h.support(s)
        return [a for a in self.h.baseline_catalogue() if all(mask[m,j] for m,j in a)]

    def outcomes(self,s,a):
        # New failures do not affect other machines' same-tick evolution or cost
        # except their own 15-unit failure charge. Integrate only relevant bits.
        following,parts=self.h.primitive_step(s,a,[False]*4)
        serving={m for m in s['assigned'] if m>=0}|{m for m,j in a}
        exposed=[m for m in range(4) if not s['failed'][m] and m not in serving and following['ages'][m]>=4]
        result=[]
        for bits in itertools.product((False,True),repeat=len(exposed)):
            after={f:list(v) for f,v in following.items()};n=sum(bits)
            for m,failed in zip(exposed,bits):
                if failed:after['failed'][m]=True;after['pending_wait'][m]=0
            result.append((.45**n*.55**(len(exposed)-n),parts['objective']+15*n,key(after)))
        self.branches+=len(result);return result

    def q(self,s,a,horizon):
        check(horizon>=1,'positive Q horizon')
        return math.fsum(p*(cost+self.value(k,horizon-1)) for p,cost,k in self.outcomes(s,a))

    def value(self,k,horizon):
        if horizon==0:return 0.
        label=(horizon,k)
        if label in self.cache:self.hits+=1;return self.cache[label]
        self.expansions+=1;check(self.expansions<=self.limit,'250000 Bellman expansion hard cap')
        if self.expansions%1000==0:self.guard()
        s=state(k);v=min(self.q(s,a,horizon) for a in self.actions(s));check(math.isfinite(v) and v>=0,'finite nonnegative Bellman value');self.cache[label]=v;return v

    def evaluate(self,s,source_pairs):
        actions=self.actions(s);values={a:self.q(s,a,4) for a in actions};joint=min(actions,key=lambda a:values[a]);mask=self.h.support(s)
        isolated=[]
        for j in range(2):
            choices=[0]+[m+1 for m in range(4) if mask[m,j]]
            isolated.append(min(choices,key=lambda token:values[((token-1,j),)] if token else values[()]))
        o=self.h.observation(s,0);iso_pairs,_,_=self.h.independent_decode(np.array(isolated),mask,ORIGINS[0],o['edges'])
        prefix=((isolated[0]-1,0),) if isolated[0] else ();occupied={m for m,j in prefix}
        choices=[0]+[m+1 for m in range(4) if mask[m,1] and m not in occupied]
        def combined(token):return tuple(sorted(prefix+(((token-1,1),) if token else ())))
        second=min(choices,key=lambda token:values[combined(token)]);conditional=combined(second)
        check(source_pairs in values,'source matching original legal support')
        return dict(isolated_tokens=isolated,conditional_tokens=[isolated[0],second],isolated_pairs=iso_pairs,conditional_pairs=conditional,joint_pairs=joint,source_pairs=source_pairs,
            isolated_q=values[iso_pairs],conditional_q=values[conditional],joint_q=values[joint],source_q=values[source_pairs],legal_matchings=len(actions))


def summarize(rows):
    groups={}
    for origin in ORIGINS:
        rr=[r for r in rows if r['origin']==origin];check(len(rr)==48,'48roots each origin')
        means={k:float(np.mean([r[k] for r in rr])) for k in ('isolated_q','conditional_q','joint_q','source_q')};check(means['isolated_q']>0 and means['source_q']>0,'nonzero pooled Q denominators')
        per={}
        for t in TRAINS:
            subset=[r for r in rr if r['train_seed']==t];iso=float(np.mean([r['isolated_q'] for r in subset]));cond=float(np.mean([r['conditional_q'] for r in subset]));per[str(t)]=dict(roots=len(subset),isolated_q=iso,conditional_q=cond,gain=1-cond/iso if iso>0 else 0.)
        groups[origin]=dict(means=means,joint_gain_vs_isolated=1-means['joint_q']/means['isolated_q'],conditional_gain_vs_isolated=1-means['conditional_q']/means['isolated_q'],source_gap_to_joint=1-means['joint_q']/means['source_q'],conditional_better_roots=sum(r['isolated_q']-r['conditional_q']>1e-8 for r in rr),per_replica=per)
    g=groups[ORIGINS[0]];screen=dict(joint_gain_at_least_half_percent=g['joint_gain_vs_isolated']>=.005,conditional_gain_at_least_half_percent=g['conditional_gain_vs_isolated']>=.005,source_headroom_at_least_half_percent=g['source_gap_to_joint']>=.005,every_replica_conditional_gain_positive=all(r['gain']>0 for r in g['per_replica'].values()),at_least_six_conditional_improvements=g['conditional_better_roots']>=6)
    return dict(classification='EXACT_H4_FIXED_STATE_DIAGNOSTIC_ONLY',origins=groups,screen=screen,conditional_q_mechanism_further_design_motivated=all(screen.values()),goal_pass=False,test_opened=False,teacher_assistance=False,updates=0,limitation='H4zero-terminal Bellman values are not H60 policy costs or health evidence. Narrow opened-state diagnostic, no CI. Exact N4 matching enumeration is not scalable policy and values must not be used as teacher labels.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);h=helpers();start=time.monotonic()
    manifest=dict(protocol=VERSION,status='RUNNING',source=str(source),verification=str(verification),source_manifest_sha256=h.sha(source/'manifest.json'),verification_manifest_sha256=h.sha(verification/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),goal_pass=False,test_opened=False,teacher_assistance=False,updates=0,new_rng_streams=0)
    write(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<600,'10minute full-audit hardstop')
    def bindings():
        for folder,hashkey,bindingkey in ((source,'source_manifest_sha256','output_bindings'),(verification,'verification_manifest_sha256','outputs')):
            check(h.sha(folder/'manifest.json')==manifest[hashkey],'input manifest unchanged')
            for name,meta in read(folder/'manifest.json')[bindingkey].items():guard();check(h.sha(folder/name)==meta['sha256'] and (folder/name).stat().st_size==meta['bytes'],'source binding '+name)
    try:
        m=read(source/'manifest.json');v=read(verification/'verification.json');check(not manifest['git_dirty'] and m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==SOURCE_COMMIT and not m['test_opened'] and m['updates']==0 and not m['future_polynomial_decoder_trial_motivated'],'frozen negative mode diagnostic')
        check(read(verification/'manifest.json')['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==manifest['source_manifest_sha256'],'verified diagnostic source');bindings()
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(__file__,out/'audit.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py');seals={name:h.sha(out/name) for name in ('protocol.md','audit.py','reference_helpers.py')}
        expected={(a,t,s,tick) for a in ORIGINS for t in TRAINS for s in SEEDS for tick in TICKS};contexts=[]
        with (source/'physical.csv').open() as f:
            for r in csv.DictReader(f):
                label=(r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']))
                if r['mode']=='greedy' and label in expected:contexts.append(dict(origin=label[0],train_seed=label[1],seed=label[2],tick=label[3],state=json.loads(r['state']),source_pairs=tuple(map(tuple,json.loads(r['pairs'])))))
        check(len(contexts)==96 and {(r['origin'],r['train_seed'],r['seed'],r['tick']) for r in contexts}==expected,'complete fixed96root grid');write(out/'contexts.json',contexts)
        engine=Exact(h,guard);results=[]
        with (out/'roots.partial.jsonl').open('w') as stream:
            for root in tqdm(contexts,desc='Exact H4 reference /96fixed roots',mininterval=1):
                guard();result=dict(**{k:v for k,v in root.items() if k not in ('state','source_pairs')},state_sha256=hashlib.sha256(json.dumps(root['state'],sort_keys=True).encode()).hexdigest(),**engine.evaluate(root['state'],root['source_pairs']));check(result['joint_q']<=min(result['source_q'],result['isolated_q'],result['conditional_q'])+1e-8,'joint minimum root bounds');results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
        write(out/'roots.json',results);summary=summarize(results);write(out/'summary.json',summary);bindings()
        check(all(h.sha(out/name)==seal for name,seal in seals.items()) and h.sha(PLAN)==seals['protocol.md'] and h.sha(Path(__file__))==seals['audit.py'] and h.sha(ROOT/'scripts/verify_broker_marginal_development.py')==seals['reference_helpers.py'],'code/protocol unchanged');manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,roots=96,unique_root_states=len({key(r['state']) for r in contexts}),bellman_expansions=engine.expansions,cache_entries=len(engine.cache),cache_hits=engine.hits,event_branches=engine.branches,conditional_q_mechanism_further_design_motivated=summary['conditional_q_mechanism_further_design_motivated'],output_bindings={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});write(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',screen=summary['screen'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start,mechanism_verdict='INCOMPLETE');write(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
