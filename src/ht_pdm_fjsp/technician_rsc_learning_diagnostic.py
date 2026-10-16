"""Frozen development-only learning, activation and input-sensitivity diagnostics."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil,subprocess,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_residual_capacity_experiment as x
from ht_pdm_fjsp.technician_completion_policy import initial_calendar

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_rsc_learning_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
SOURCE_COMMIT='b44f114a8bc44ca2fefdfcd4eadf18d1e0f5322d'
STEPS=(0,122880,245760,368640,491520)


def first_log(model,batch):
    base=model.encoder(model.raw_pairs(batch));calendar,planned=initial_calendar(batch)
    return model.conditionals(batch,base,calendar,planned,
        torch.zeros(len(base),dtype=torch.long),0,
        torch.zeros(base.shape[:2],dtype=torch.bool))[0]


def inspect_policy(model,observations,corpus):
    """Local network derivatives; no environment step, optimizer or model.grad."""
    before=x.learner.parameter_digest(model);algorithm=model.algorithm
    batch=x.learner.tensors(observations)
    batch['edges']=batch['edges'].detach().requires_grad_(True)
    inputs=[];saturations={};hidden=[];embedded=[];handles=[]
    def capacity_input(module,args):
        value=args[0].detach().clone().requires_grad_(True);inputs.append(value)
        return (value,)
    handles.append(model.capacity_encoder.register_forward_pre_hook(capacity_input))
    handles.append(model.capacity_encoder.register_forward_hook(lambda m,i,o:embedded.append(o.detach())))
    handles.append(model.bid[3].register_forward_hook(lambda m,i,o:hidden.append(o.detach())))
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Tanh) and not name.startswith('value_'):
            def hook(m,i,o,key=name):saturations[key]=float((o.detach().abs()>.99).float().mean())
            handles.append(module.register_forward_hook(hook))
    try:
        log=first_log(model,batch)
        message_grad,edge_grad=torch.autograd.grad((log[:,2]-log[:,1]).sum(),(inputs[0],batch['edges']))
        probability=log.detach().exp();h=hidden[0]
        result=dict(probability_wait=float(probability[:,0].mean()),probability_machine1=float(probability[:,1].mean()),
            probability_machine2=float(probability[:,2].mean()),
            probability_critical_given_nonwait=float((probability[:,2]/probability[:,1:].sum(1).clamp_min(1e-15)).mean()),
            first_slot_entropy=float(-(probability*log.detach()).sum(1).mean()),
            capacity_input_logodds_gradient_l2=float(message_grad.flatten(1).norm(dim=1).mean()),
            own_service_logodds_gradient_per_tick_l2=float((edge_grad[:,:2,0,1]/12).norm(dim=1).mean()),
            bid_second_hidden_machine12_l2=float((h[:,0]-h[:,1]).norm(dim=1).mean()),
            capacity_embedding_machine12_l2=float((embedded[0][:,0]-embedded[0][:,1]).norm(dim=1).mean()),
            actor_tanh_saturation=saturations,
            capacity_output_weight_l2=float(model.capacity_encoder[-1].weight.detach().norm()))
    finally:
        for handle in handles:handle.remove()
    try:
        with torch.no_grad():
            model.algorithm='no_capacity_ppo';zero=first_log(model,batch).exp()
            result['same_weight_zero_message_tv']=float((probability-zero).abs().sum(1).mean()/2)
            model.algorithm=algorithm
            fixed=x.learner.tensors(corpus);actual=first_log(model,fixed).exp()
            model.algorithm='no_capacity_ppo';zero=first_log(model,fixed).exp()
            result['initial_corpus_mean_zero_message_tv']=float((actual-zero).abs().sum(1).mean()/2)
    finally:model.algorithm=algorithm
    if x.learner.parameter_digest(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('diagnostic mutated weights or accumulated gradients')
    if not all(math.isfinite(v) for v in result.values() if isinstance(v,float)):
        raise RuntimeError('nonfinite diagnostic')
    return result


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic()
    def guard():
        if time.monotonic()-start>600:raise TimeoutError('diagnostic10min hardstop')
    def read(name):return json.loads((source/name).read_text())
    m=dict(protocol=VERSION,status='RUNNING',classification='POSTHOC_DEVELOPMENT_DIAGNOSTIC',
        source=str(source),source_manifest_sha256=x.sha(source/'manifest.json'),
        test_opened=False,goal_pass=False,training_steps=0,new_physical_episodes=0,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',m)
    try:
        if m['git']['dirty']:raise RuntimeError('commit/push before diagnostic')
        original=read('manifest.json');cfg=read('resolved_config.json');s=read('summary.json')
        if (original['status']!='COMPLETED' or original['audit_status']!='PASS' or original['test_opened']
            or original['git']['commit']!=SOURCE_COMMIT or cfg!=x.settings('development')):
            raise RuntimeError('frozen source prerequisite')
        if len(original['output_bindings'])!=519:raise RuntimeError('source binding inventory')
        for name,meta in tqdm(original['output_bindings'].items(),desc='Verify frozen RSC source'):
            guard();p=source/name
            if x.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise RuntimeError('source binding '+name)
        for name,h in original['source_seal'].items():
            if x.sha(source/name)!=h:raise RuntimeError('source seal')
            if name.startswith('source_snapshot/') and x.sha(Path(__file__).parent/Path(name).name)!=h:
                raise RuntimeError('learning source changed '+name)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(__file__,out/'diagnostic.py')
        curves=defaultdict(list);rows=[]
        with (source/'development_episodes.csv').open() as f:
            for row in csv.DictReader(f):
                curves[row['method'],int(row['train_seed']),int(row['checkpoint_steps'])].append(row)
        expected={(a,t,step) for a in x.arms(4) for t in cfg['train_seeds'] for step in STEPS}
        if set(curves)!=expected or any(len(v)!=32 for v in curves.values()):raise RuntimeError('curve grid')
        for (a,t,step),records in sorted(curves.items()):
            rows.append(dict(method=a,train_seed=t,steps=step,**{key:float(np.mean([float(r[key]) for r in records]))
                for key in ('normalized_cost','availability','normalized_pending','normalized_failures')}))
        x.write_csv(out/'learning_curves.csv',rows)
        observations=[r['observation'] for r in x.mechanism_states('development')];probes=[]
        for a,t,step in tqdm(list(sorted(k for k in expected if k[0] in x.CORE_ARMS)),desc='Frozen actor sensitivity'):
            guard();p=source/f'N4/{a}/{t}/checkpoints/step_{step:09d}.pt'
            model=x.load_model(p,4,cfg,t,a,step)
            fixed=source/f'N4/residual_capacity_ppo/{t}/checkpoints/latency_observations_000000000.npz'
            with np.load(fixed,allow_pickle=False) as archive:
                corpus=[{k:archive[k][i].copy() for k in archive.files} for i in range(48)]
            r=inspect_policy(model,observations,corpus);r['actor_tanh_saturation']=json.dumps(r['actor_tanh_saturation'],sort_keys=True)
            probes.append(dict(method=a,train_seed=t,steps=step,checkpoint_sha256=x.sha(p),initial_corpus_sha256=x.sha(fixed),**r))
        x.write_csv(out/'actor_sensitivity.csv',probes)
        updates={}
        for a in x.CORE_ARMS:
            for t in cfg['train_seeds']:
                with (source/f'N4/{a}/{t}/training_progress.csv').open() as f:data=list(csv.DictReader(f))
                if len(data)!=160:raise RuntimeError('update grid')
                updates[f'{a}:{t}']={key:dict(mean=float(np.mean([float(r[key]) for r in data])),
                    last=float(data[-1][key]),positive_updates=sum(float(r[key])>0 for r in data))
                    for key in ('actor_update_norm','critic_update_norm')}
        x.write_json(out/'update_activity.json',updates)
        summary=dict(classification=m['classification'],goal_pass=False,test_opened=False,
            actor_probe_models=len(probes),checkpoint_curves=len(rows),source_output_bindings_verified=519,
            primary_failed_run_reinterpreted=False,
            candidate_final_selected_all=all(s['selected_steps'][f'4:residual_capacity_ppo:{t}']==491520 for t in cfg['train_seeds']))
        x.write_json(out/'summary.json',summary)
        lines=['# Frozen RSC learning diagnostic','',
            'POSTHOC DEVELOPMENT ONLY. No training, new physical episode or scientific test. The original failed verdict is preserved.','',
            '| Method | Cost/m at368640 | Cost/m at491520 | Last-quarter gain |','|---|---:|---:|---:|']
        for a in x.arms(4):
            costs={step:float(np.mean([r['normalized_cost'] for r in rows if r['method']==a and r['steps']==step])) for step in (368640,491520)}
            lines.append(f"| {a} | {costs[368640]:.6f} | {costs[491520]:.6f} | {1-costs[491520]/costs[368640]:+.6%} |")
        lines+=['','| Core arm | Probe zero-message TV | Capacity-input gradient norm | Bid-layer2 saturation |','|---|---:|---:|---:|']
        for a in x.CORE_ARMS:
            panel=[r for r in probes if r['method']==a and r['steps']==491520]
            values=[float(np.mean([r[k] for r in panel])) for k in ('same_weight_zero_message_tv','capacity_input_logodds_gradient_l2')]
            saturated=float(np.mean([json.loads(r['actor_tanh_saturation'])['bid.3'] for r in panel]))
            lines.append(f"| {a} | {values[0]:.8f} | {values[1]:.8f} | {saturated:.8f} |")
        lines+=['','These local neural derivatives are not physical-service derivatives or causal evidence for an architecture change. Fixed-probe contrasts are descriptive.','',
            'All three candidate replicas selecting the last checkpoint plus its positive final-quarter curve can motivate one prelocked2x-budget test. Every method must receive the same budget, fresh random weights/panels and unchanged architecture/objective. Tuning alone is not the scientific contribution; the15 quality/health/mechanism limits remain.']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        # Source inventory was sealed; reread hashes after all probes to rule out writes.
        for name,meta in original['output_bindings'].items():
            guard()
            if x.sha(source/name)!=meta['sha256']:raise RuntimeError('source changed during diagnostic')
        if x.sha(source/'manifest.json')!=m['source_manifest_sha256']:raise RuntimeError('source manifest changed')
        m.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,
                 output_bindings={p.name:dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        x.write_json(out/'manifest.json',m);print(json.dumps(dict(output=str(out),**summary)));return m
    except Exception as exc:
        m.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        x.write_json(out/'manifest.json',m);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.source,a.output_dir)

if __name__=='__main__':main()
