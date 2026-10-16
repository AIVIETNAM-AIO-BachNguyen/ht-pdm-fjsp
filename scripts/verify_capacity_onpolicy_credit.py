"""Independently verify frozen credit arrays and gradient/episode arithmetic."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))
def near(a,b):assert abs(a-b)<=1e-8*max(1,abs(b)),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);m=read(source/'manifest.json');s=read(source/'summary.json')
    assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened'] and m['training_steps']==0
    for name,meta in m['output_bindings'].items():assert sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes']
    counts={}
    for family,meta in m['sources'].items():
        p=Path(meta['path']);assert sha(p/'manifest.json')==meta['manifest_sha256'];sm=read(p/'manifest.json')
        for name,b in sm['output_bindings'].items():assert sha(p/name)==b['sha256'] and (p/name).stat().st_size==b['bytes']
        counts[family]=len(sm['output_bindings'])
    populations={};allchecks={f:{} for f in m['sources']};all_ep=[];models=set()
    for r in s['records']:
        key=r['family'],r['method'],r['train_seed'];assert key not in models;models.add(key)
        directory=source/r['family']/r['method']/str(r['train_seed']);ep=csvrows(directory/'episodes.csv');dec=csvrows(directory/'decisions.csv')
        assert len(ep)==64 and len(dec)==3072;all_ep.extend(ep)
        with np.load(directory/'credit_arrays.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        reward=data['reward'];value=data['value'];ret=np.zeros((48,64));gae=np.zeros_like(ret)
        for t in range(48):ret[t]=np.sum(reward[t:],axis=0)
        delta=reward+np.concatenate([value[1:],np.zeros((1,64))])-value
        for t in range(48):gae[t]=np.sum(delta[t:]*.95**np.arange(48-t)[:,None],axis=0)
        np.testing.assert_allclose(ret,data['returns'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(ret-value,data['mc_advantage'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(gae,data['gae_advantage'],atol=1e-10,rtol=0)
        for row in dec:
            t=int(row['tick']);i=int(row['episode']);near(float(row['reward']),reward[t,i]);near(float(row['value']),value[t,i])
        with np.load(directory/'observations.npz',allow_pickle=False) as f:
            actions=f['actions'];mask=f['pair_mask'];joint=f['joint_mask'] if 'joint_mask' in f.files else None
        for row,act,support in zip(dec,actions,mask):
            assert np.array_equal(np.asarray(json.loads(row['action'])),act)
            if np.ndim(act)==0:assert joint[int(row['tick'])*64+int(row['episode']),int(act)]
            else:
                used=[]
                for j,v in enumerate(act[:2]):
                    if v:assert support[int(v)-1,j] and int(v) not in used;used.append(int(v))
        for row in ep:
            i=int(row['episode']);near(-20*reward[:,i].sum(),float(row['extended_cost']))
            assert row['physical_ticks']=='60' and float(row['invalid_assignments'])==0
            previous=populations.setdefault(row['seed'],row['population_sha256']);assert previous==row['population_sha256']
        error=value-ret;ev=1-float(np.var(error))/float(np.var(ret));bias=abs(float(error.mean()))/float(ret.std())
        near(ev,r['critic_explained_variance']);near(bias,r['absolute_mean_error_in_return_sd'])
        with np.load(directory/'block_gradients.npz',allow_pickle=False) as f:grads={k:f[k].copy() for k in f.files}
        dispersion={};means={};norms={}
        for label,g in grads.items():
            assert g.shape[0]==4 and np.isfinite(g).all();mean=g.sum(0)/4;means[label]=mean;norms[label]=float(np.sum(mean**2))
            dispersion[label]=float(np.sum((g-mean)**2)/4)/max(norms[label],1e-20)
            near(dispersion[label],r['gradient'][label]['dispersion']);near(norms[label],r['gradient'][label]['mean_gradient_squared_norm'])
        ratio=dispersion['gae']/max(dispersion['mc'],1e-20);denom=float(np.linalg.norm(means['mc'])*np.linalg.norm(means['gae']))
        cosine=float(np.dot(means['mc'],means['gae'])/denom) if denom>1e-20 else 0
        near(ratio,r['gradient']['dispersion_ratio']);near(cosine,r['gradient']['mean_gradient_cosine'])
        if r['method']=='residual_capacity_ppo':allchecks[r['family']][str(r['train_seed'])]=dict(critic_explained_variance=ev>=.5,prediction_mean_error=bias<=.1,gradient_dispersion=ratio<=.75,gradient_direction=cosine>=.8,nonzero_mc_gradient=norms['mc']>1e-20)
    assert len(models)==21 and len(all_ep)==1344 and len(populations)==64
    for family,checks in allchecks.items():assert len(checks)==3 and checks==s['eligibility']['families'][family]['checks'] and all(all(v.values()) for v in checks.values())==s['eligibility']['families'][family]['eligible']
    result=dict(status='PASS',source=str(source),source_manifest_sha256=sha(source/'manifest.json'),bindings_verified=len(m['output_bindings']),original_source_bindings=counts,models=21,episodes=1344,decisions=64512,gradient_blocks=168,gae_trial_motivated=False,limitation='Independent stored-data arithmetic and support checks; does not independently rerun neural gradients or physical trajectories.')
    shutil.copy2(__file__,out/'verifier.py');(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
