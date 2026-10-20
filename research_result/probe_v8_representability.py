"""DEV-only exact local-edge collision census and supervised capacity check."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_requests import state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'research_result/v8_headroom_dev_20261020T052426Z'
PROTOCOL = ROOT / 'research_result/v8_representability_probe_plan_vi.md'
SOURCES = [ROOT / x for x in (
    'research_result/probe_v8_representability.py',
    'research_result/v8_representability_probe_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_framework_v6_policy.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
)]
ACTIONS = catalogue()
INCIDENCE = np.zeros((len(ACTIONS), 8), np.float32)
for ai, action in enumerate(ACTIONS):
    for machine, worker in action:
        INCIDENCE[ai, machine * 2 + worker] = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(row):
    config = configs()[row['case']]
    state = state_from_dict(row['original'])
    admitted = submit_requests(config, state, row['tick'], tuple(row['submissions']))
    obs = vector_observation(config, admitted, row['tick'])
    scores = {int(k): float(v) for k, v in row['scores'].items()}
    legal = tuple(int(a) for a in row['legal_actions'])
    if set(scores) != set(legal) or len(legal) < 2:
        raise AssertionError('invalid legal/root score set')
    return obs, legal, scores


def local_edge_bytes(obs, machine, worker):
    edge = machine * 2 + worker
    return np.concatenate((obs[:2], obs[2 + 8*machine: 10 + 8*machine],
                           obs[34 + 3*worker: 36 + 3*worker],
                           obs[40 + 2*edge: 42 + 2*edge])).astype(np.float32).tobytes()


def collision_key(obs, left, right):
    la, ra = set(ACTIONS[left]), set(ACTIONS[right])
    signed = tuple(sorted([(1, m, j, local_edge_bytes(obs, m, j)) for m, j in la-ra] +
                          [(-1, m, j, local_edge_bytes(obs, m, j)) for m, j in ra-la]))
    return left, right, signed


def opposite_context_pairs(plus, minus):
    return [(p,n) for p in plus for n in minus if p[5] != n[5]]


def collisions(items, timeout):
    groups = defaultdict(list)
    pair_records = 0
    for row in tqdm(items, desc='Exact local-edge pairs', unit='state'):
        timeout()
        obs, legal, scores = reconstruct(row)
        for i, a in enumerate(legal):
            for b in legal[i+1:]:
                left, right = sorted((a,b))
                margin = scores[right] - scores[left]
                if margin == 0:
                    continue
                groups[collision_key(obs, left, right)].append((int(row['state_id']),
                    row['case'], row['source_arm'], int(row['train_seed']), margin, obs.tobytes()))
                pair_records += 1
    result = dict(pair_records=pair_records, collision_key_count=len(groups), thresholds={})
    for threshold in (1,5,10):
        keys = state_pairs = 0
        covered = set()
        by_case = defaultdict(int)
        by_source_seed = defaultdict(int)
        examples = []
        for key, values in groups.items():
            plus = [x for x in values if x[4] >= threshold]
            minus = [x for x in values if x[4] <= -threshold]
            valid = opposite_context_pairs(plus, minus)
            if not valid:
                continue
            keys += 1
            state_pairs += len(valid)
            involved = {x for pair in valid for x in pair}
            covered.update(x[0] for x in involved)
            for x in involved:
                by_case[x[1]] += 1
                by_source_seed[f'{x[2]}/{x[3]}'] += 1
            if len(examples) < 12:
                examples.append(dict(left_action=key[0], right_action=key[1],
                                     state_a=valid[0][0][0], state_b=valid[0][1][0],
                                     margins=[valid[0][0][4],valid[0][1][4]]))
        result['thresholds'][str(threshold)] = dict(collision_keys=keys,
            opposite_preference_record_pairs=state_pairs, unique_states=len(covered),
            states=sorted(covered), by_case=dict(sorted(by_case.items())),
            by_source_seed=dict(sorted(by_source_seed.items())), examples=examples)
    return result


class JointScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64,64), nn.Tanh(),
                                 nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
        self.register_buffer('incidence', torch.tensor(INCIDENCE))
    def forward(self, obs):
        n = obs.shape[0]
        x = torch.cat((obs[:,None,:].expand(n,len(ACTIONS),56),
                       self.incidence[None,:,:].expand(n,len(ACTIONS),8)), dim=2)
        return self.net(x).squeeze(-1)


class FlatScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(56,64), nn.Tanh(),
                                 nn.Linear(64,64), nn.Tanh(), nn.Linear(64,len(ACTIONS)))
    def forward(self, obs):
        return self.net(obs)


def dataset(items):
    obs, mask, target, records = [], [], [], []
    for row in items:
        x, legal, scores = reconstruct(row)
        m = np.zeros(len(ACTIONS), np.float32)
        m[list(legal)] = 1
        t = np.zeros(len(ACTIONS), np.float32)
        mean = np.mean([scores[a] for a in legal])
        for a in legal:
            t[a] = -(scores[a]-mean)
        obs.append(x); mask.append(m); target.append(t); records.append(row)
    return (torch.tensor(np.asarray(obs)),torch.tensor(np.asarray(mask)),
            torch.tensor(np.asarray(target)),records)


def split(items):
    buckets = {'train': [], 'selection': [], 'probe': []}
    for row in items:
        seed = int(row['physical_seed'])
        if 69100000 <= seed <= 69100019: key='train'
        elif 69100020 <= seed <= 69100023: key='selection'
        elif 69100024 <= seed <= 69100031: key='probe'
        else: raise AssertionError('unregistered physical seed')
        buckets[key].append(row)
    return buckets


def masked_mse(model, data):
    obs, mask, target, _ = data
    logits = model(obs)
    return (((logits-target)**2*mask).sum(dim=1)/mask.sum(dim=1)).mean()


def metrics(model, data):
    obs, mask, _, rows = data
    with torch.no_grad():
        scores = model(obs).cpu().numpy()
        mse = float(masked_mse(model,data))
    regrets=[]; risk=[]; original=[]; large=0; exact=0
    for i,row in enumerate(rows):
        legal = [int(a) for a in row['legal_actions']]
        pred = max(legal,key=lambda a:(float(scores[i,a]),-a))
        values = {int(a):float(v) for a,v in row['scores'].items()}
        best = values[int(row['planner_action'])]
        regret = values[pred]-best
        regrets.append(regret)
        risk.append(values[int(row['risk_action'])]-best)
        original.append(values[int(row['source_action'])]-best)
        large += regret >= 5
        exact += pred == int(row['planner_action'])
    return dict(mse=mse,states=len(rows),mean_root_regret=float(np.mean(regrets)),
                gap_ge_5=large,planner_action_match=exact,
                risk_mean_root_regret=float(np.mean(risk)),
                source_mean_root_regret=float(np.mean(original)))


def fit(items, profile, timeout):
    buckets=split(items)
    if profile=='smoke':
        # Keep at least one record per physical split; smoke checks the entire pipeline.
        buckets={k:v[:min(12,len(v))] for k,v in buckets.items()}
    if any(not v for v in buckets.values()):
        raise AssertionError('empty physical-seed split')
    data={k:dataset(v) for k,v in buckets.items()}
    starts=(70200000,) if profile=='smoke' else tuple(range(70200000,70200005))
    epochs=2 if profile=='smoke' else 100
    result=dict(split_counts={k:len(v) for k,v in buckets.items()},starts=list(starts),
                epochs=epochs,models={})
    for name,cls in [('edge',lambda:EdgeSharingActor('shared_edge')),
                     ('flat',FlatScorer),('joint',JointScorer)]:
        result['models'][name]=[]
        for seed in tqdm(starts,desc=f'Supervised capacity {name}',unit='start'):
            timeout(); torch.manual_seed(seed); np.random.seed(seed)
            model=cls(); opt=torch.optim.Adam(model.parameters(),lr=.001)
            best=float('inf'); chosen=None; chosen_epoch=-1
            for epoch in range(1,epochs+1):
                timeout(); model.train(); opt.zero_grad()
                loss=masked_mse(model,data['train']); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad(): val=float(masked_mse(model,data['selection']))
                if val < best:
                    best=val; chosen={k:v.detach().clone() for k,v in model.state_dict().items()}; chosen_epoch=epoch
            model.load_state_dict(chosen)
            result['models'][name].append(dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                selected_epoch=chosen_epoch,train=metrics(model,data['train']),
                selection=metrics(model,data['selection']),probe=metrics(model,data['probe'])))
    return result


def run(profile,out):
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    begin=time.monotonic(); limit=begin+(120 if profile=='smoke' else 900)
    def timeout():
        if time.monotonic()>limit: raise TimeoutError('representability probe timeout')
    manifest=dict(status='RUNNING',profile=profile,started_utc=datetime.now(timezone.utc).isoformat(),
        invocation=[sys.executable,*sys.argv],cwd=str(ROOT),runtime=dict(platform=platform.platform(),
            python=platform.python_version(),torch=torch.__version__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        input_hashes={name:sha(SOURCE/name) for name in ('states.jsonl','manifest.json','summary.json')})
    def save(name,obj): (out/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    save('manifest.json',manifest)
    try:
        source_manifest=json.loads((SOURCE/'manifest.json').read_text())
        if source_manifest['status']!='COMPLETED': raise AssertionError('source incomplete')
        items=[json.loads(line) for line in (SOURCE/'states.jsonl').open()]
        if len(items)!=720 or len({x['state_id'] for x in items})!=720: raise AssertionError('state count')
        if profile=='smoke':
            ids=set(range(80))
            for seed in (69100020,69100024):
                ids.update(x['state_id'] for x in items if x['physical_seed']==seed)
            items=[x for x in items if x['state_id'] in ids]
        manifest['input_states']=len(items); save('manifest.json',manifest)
        census=collisions(items,timeout); save('collisions.json',census)
        capacity=fit(items,profile,timeout); save('capacity.json',capacity)
        if {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}!=manifest['source_hashes']:
            raise RuntimeError('source changed during probe')
        manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-begin,
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        save('manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.monotonic()-begin)
        save('manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','dev'),required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    print(json.dumps({'status':'COMPLETED','output_dir':str(run(args.profile,args.output_dir))}))

if __name__=='__main__': main()
