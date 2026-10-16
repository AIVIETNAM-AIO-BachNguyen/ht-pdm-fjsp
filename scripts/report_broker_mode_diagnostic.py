"""Descriptive report of a completed, independently verified frozen diagnostic."""
import argparse,csv,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def check(ok,label):
    if not ok:raise AssertionError(label)


def run(source,verification,out):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    sm=read(source/'manifest.json');v=read(verification/'verification.json');check(sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json'),'verified source binding')
    summary=read(source/'summary.json');rows=list(csv.DictReader((source/'episodes.csv').open()));arms=list(summary['means']['greedy']);mechanism={}
    for a in arms:
        for mode in ('greedy','matching_mode'):
            rr=[r for r in rows if r['method']==a and r['mode']==mode];mechanism[f'{a}:{mode}']=dict(same_state_mismatches=sum(int(r['same_state_mismatches']) for r in rr),decisions=48*len(rr),executed_rejections=sum(int(r['executed_rejections']) for r in rr),mean_expected_latent_rejections_per_decision=sum(float(r['expected_latent_rejections']) for r in rr)/(48*len(rr)))
    # Exploratory current-state support counts only; no fallback action is executed.
    candidate='broker_marginal_ppo';counts=Counter();fallback_durations=Counter();pairs_by_method=defaultdict(Counter)
    with (source/'physical.csv').open() as pf,(source/'policy.csv').open() as qf:
        policies=iter(csv.DictReader(qf))
        for r in tqdm(csv.DictReader(pf),total=184320,desc='Descriptive recorded-state counts'):
            if int(r['tick'])>=48:continue
            q=next(policies);check(all(r[k]==q[k] for k in ('method','train_seed','seed','mode','tick')),'physical/policy join');pairs=json.loads(r['pairs']);state=json.loads(r['state'])
            if r['mode']=='greedy':pairs_by_method[r['method']].update((m,j) for m,j in pairs)
            if r['method']!=candidate or r['mode']!='greedy':continue
            counts['decisions']+=1;selected={m for m,j in pairs};used={j for m,j in pairs};busy={m for m in state['assigned'] if m>=0};eligible={m for m in range(4) if m not in busy and (state['failed'][m] or state['ages'][m]>=3)}
            submitted=json.loads(q['submitted']);rejected=[j for j,t in enumerate(submitted) if t and j not in used]
            counts['rejected_proposals']+=len(rejected)
            for j in rejected:
                alternate=eligible-selected
                if not alternate:continue
                counts['rejections_with_feasible_alternative']+=1;counts['rejections_with_failed_alternative']+=int(any(state['failed'][m] for m in alternate));fallback_durations[str(min([[1,2],[1,4],[2,1],[4,1]][m][j] for m in alternate))]+=1
    components={}
    for a in (candidate,'broker_proposal_ppo','ppo'):
        components[a]={mode:{k:float(np.mean([float(r[k])/4 for r in rows if r['method']==a and r['mode']==mode])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')} for mode in ('greedy','matching_mode')}
    analysis=dict(classification='DESCRIPTIVE_DIAGNOSTIC_DEV_ONLY',source=str(source),verification=str(verification),summary=summary,mechanism=mechanism,exploratory_recorded_state_counts=dict(counts),exploratory_minimum_alternative_duration_counts=dict(fallback_durations),greedy_pair_counts={a:{f'm{m}_t{j}':n for (m,j),n in values.items()} for a,values in pairs_by_method.items()},h60_cost_components_per_machine=components,limitation='Alternative support counts do not establish beneficial reassignment, causal opportunity cost or a learning hypothesis. No new action/state, CI or scientific-test claim.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2))
    names=['Physical-q','Proposal-p','ID physical-q','ID proposal-p','PPO','PPO norm','MAPPO local','MAPPO full'];fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');y=np.arange(8);ppo=summary['means']['greedy']['ppo']['normalized_cost']
    for mode,offset,color in (('greedy',-.17,'#607D8B'),('matching_mode',.17,'#1976D2')):
        delta=[100*(summary['means'][mode][a]['normalized_cost']/ppo-1) for a in arms];axes[0].barh(y+offset,delta,height=.30,label=mode,color=color)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].axvline(0,color='black',lw=.7);axes[0].set_xlabel('Mean H60 cost relative to original PPO (%)');axes[0].set_title('All methods receive both decoders');axes[0].legend(fontsize=8)
    rep=summary['candidate_per_replica'];gains=[100*r['gain'] for r in rep.values()];axes[1].bar(range(3),gains,color='#1976D2');axes[1].axhline(.5,color='#C62828',ls='--',label='Locked pooled practical threshold: 0.5%');axes[1].set_xticks(range(3),['Replica 1','Replica 2','Replica 3']);axes[1].set_ylabel('Own mode-vs-greedy cost gain (%)');axes[1].set_title('Candidate: pooled gain 0.0233%');axes[1].legend(fontsize=8,loc='upper right')
    for i,value in enumerate(gains):axes[1].text(i,value+.009,f'{value:.4f}%',ha='center',fontsize=9)
    fig.suptitle('Frozen diagnostic: decoder remedy rejected; descriptive means, no CI',fontsize=12);fig.savefig(out/'mode_diagnostic.png',dpi=170);plt.close(fig)
    shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
