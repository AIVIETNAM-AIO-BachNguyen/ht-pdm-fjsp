"""Matched true/shuffled pair regression, checkpointed on held-out pair MSE."""
import copy
import hashlib
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import VERSION
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead,CONTRACT
from ht_pdm_fjsp.maintenance_opportunity_cost import write_json


def predictions(head,data,rows=None):
    pairs=data["pairs"] if rows is None else data["pairs"][rows]
    features={k:data[k][pairs[:,0]] for k in ("raw_pairs","context")}
    scores=head(features);ids=torch.arange(len(pairs))
    return scores[ids,pairs[:,2]]-scores[ids,pairs[:,1]]


@torch.no_grad()
def mse(head,data):return float((predictions(head,data)-data["targets"]).square().mean())


def shuffled_permutation(conditions,seed):
    rng=np.random.default_rng(role_seed(seed,f"{VERSION}:shuffle"));perm=np.arange(len(conditions))
    for cond in sorted(set(conditions.tolist())):
        ids=np.flatnonzero(conditions.numpy()==cond);perm[ids]=rng.permutation(ids)
    if sorted(perm.tolist())!=list(range(len(conditions))) or not np.array_equal(conditions.numpy(),conditions.numpy()[perm]):raise RuntimeError("shuffle contract")
    return torch.tensor(perm,dtype=torch.long)


def fit(output,cfg,train_seed,source_seed,source_hash,train,dev,journal,stats):
    torch.manual_seed(role_seed(train_seed,f"{VERSION}:initialization"))
    first=RelativeValueHead(cfg["hidden"]);heads={"true_value":first,"shuffled_value":copy.deepcopy(first)}
    if any(not torch.equal(v,heads["shuffled_value"].state_dict()[k]) for k,v in first.state_dict().items()):raise RuntimeError("initial weights mismatch")
    perm=shuffled_permutation(train["conditions"],train_seed)
    targets={"true_value":train["targets"],"shuffled_value":train["targets"][perm]}
    path=output/"datasets"/f"shuffle_{source_seed}.pt";torch.save(perm,path)
    meta=dict(permutation=perm.tolist(),conditions=train["conditions"].tolist(),
        sha256=hashlib.sha256(perm.numpy().tobytes()).hexdigest(),identity_fraction=float((perm==torch.arange(len(perm))).float().mean()))
    write_json(output/"datasets"/f"shuffle_{source_seed}.json",meta)
    optimizers={name:torch.optim.Adam(head.parameters(),lr=cfg["learning_rate"]) for name,head in heads.items()}
    generator=torch.Generator().manual_seed(role_seed(train_seed,f"{VERSION}:minibatches"))
    records={name:[] for name in heads};init_hash=hashlib.sha256(b"".join(v.numpy().tobytes() for v in first.state_dict().values())).hexdigest()
    for epoch in tqdm(range(1,cfg["epochs"]+1),desc=f"Train paired value heads {source_seed}",unit="epoch"):
        order=torch.randperm(len(train["pairs"]),generator=generator);losses={name:[] for name in heads}
        for start in range(0,len(order),cfg["minibatch"]):
            ids=order[start:start+cfg["minibatch"]]
            for name,head in heads.items():
                head.train();pred=predictions(head,train,ids)
                loss=torch.nn.functional.smooth_l1_loss(pred,targets[name][ids]);optimizers[name].zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),cfg["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError("nonfinite training")
                optimizers[name].step();losses[name].append((float(loss.detach()),len(ids)));stats["optimizer_steps"]+=1
        for name,head in heads.items():
            head.eval();row=dict(source_seed=source_seed,train_seed=train_seed,controller=name,epoch=epoch,
                train_smooth_l1=sum(v*n for v,n in losses[name])/len(train["pairs"]),true_train_mse=mse(head,train),true_development_mse=mse(head,dev),labels=len(train["pairs"]),matched_initial_sha=init_hash)
            journal.add(row)
            if epoch in cfg["checkpoints"]:
                p=output/name/f"source_{source_seed}"/f"epoch_{epoch}.pt";p.parent.mkdir(parents=True,exist_ok=True)
                payload=dict(protocol=VERSION,feature_contract=CONTRACT,settings=cfg,source_seed=source_seed,source_sha256=source_hash,train_seed=train_seed,epoch=epoch,state_dict=head.state_dict())
                torch.save(payload,p);restored=RelativeValueHead(cfg["hidden"]);restored.load_state_dict(torch.load(p,weights_only=True)["state_dict"])
                if any(not torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items()) or not torch.equal(predictions(head,dev),predictions(restored,dev)):raise RuntimeError("head checkpoint roundtrip")
                records[name].append({**row,"checkpoint":p.relative_to(output).as_posix()})
    selections={};result={}
    for name in heads:
        selected=min(records[name],key=lambda r:(r["true_development_mse"],r["epoch"]))
        payload=torch.load(output/selected["checkpoint"],weights_only=True);head=RelativeValueHead(cfg["hidden"]);head.load_state_dict(payload["state_dict"])
        if payload["source_sha256"]!=source_hash or payload["feature_contract"]!=CONTRACT:raise RuntimeError("head source binding")
        result[name]=head.eval();selections[name]=dict(selected=selected,candidates=records[name],true_dev_labels_only=True)
    return result,selections
