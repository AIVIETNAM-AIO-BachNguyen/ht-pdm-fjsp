"""Exact latent joint-trace PPO policy over bounded bilateral exchanges."""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot

from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors
from ht_pdm_fjsp.technician_completion_policy import initial_calendar, calendar_embeddings
from ht_pdm_fjsp.technician_exchange_v2_env import CORE_ARMS, BILATERAL_ARMS, value_unit


def intent_view(matching, technician, independent):
    if not independent:
        return matching
    view = torch.zeros_like(matching)
    rows = torch.arange(len(matching), device=matching.device)
    view[rows, technician] = matching[rows, technician]
    return view


def provisional_calendar(batch, matching):
    n = batch['machines'].shape[1]
    calendar, planned = initial_calendar(batch)
    intent = one_hot(matching, n+1)[..., 1:].transpose(1, 2).to(calendar.dtype)
    active = intent.any(2)
    duration = (batch['edges'][..., 1]*intent).sum(2)
    restoration = (batch['edges'][..., 2]*intent).sum(2)
    completion = torch.floor(batch['machines'][..., 0]*8*(1-restoration))/8
    return torch.stack([torch.where(active, 1., calendar[..., 0]),
        torch.where(active, duration, calendar[..., 1]),
        torch.where(active, completion, calendar[..., 2])], -1), planned+intent, intent


def move_support(batch, matching, technician, repair, arm, previous_owner=None):
    b, n, k = batch['edges'].shape[:3]
    view = intent_view(matching, technician, arm == 'independent_ppo')
    intent = one_hot(view, n+1)[..., 1:].transpose(1, 2).bool()
    has_owner, owners = intent.any(2), intent.long().argmax(2)
    own = matching[:, technician]
    legal = batch['pair_mask'][:, :, technician].clone()
    if not repair:
        legal &= ~has_owner
        drop = torch.zeros(b, dtype=torch.bool, device=matching.device)
    else:
        legal &= (torch.arange(n, device=matching.device)[None]+1 != own[:, None])
        if arm in BILATERAL_ARMS:
            rows = torch.arange(b, device=matching.device)[:, None]
            reciprocal = batch['pair_mask'][rows, (own-1).clamp(min=0)[:, None], owners]
            legal &= ~has_owner | (own[:, None] == 0) | reciprocal
            if arm == 'exchange_ppo':
                legal &= ~has_owner | (owners > technician)
            elif arm == 'inverse_mask_ppo':
                if previous_owner is None:
                    raise ValueError('inverse support requires recorded preceding-slot state')
                legal &= ~((previous_owner[:, None] == technician) & has_owner & (owners == technician-1))
        elif arm != 'independent_ppo':
            legal &= ~has_owner
        drop = own > 0
        if arm == 'construct_only_ppo':
            legal[:] = False
            drop[:] = False
    return torch.cat([torch.ones((b, 1), dtype=torch.bool, device=matching.device), legal, drop[:, None]], 1)


def apply_move(matching, action, technician, repair, arm, n):
    result = matching.clone()
    if not repair:
        result[:, technician] = action
        return result
    target = (action > 0) & (action <= n)
    if arm in BILATERAL_ARMS:
        claimed = (matching == action[:, None]) & target[:, None]
        rows = torch.where(claimed.any(1))[0]
        owner = claimed.long().argmax(1)
        result[rows, owner[rows]] = matching[rows, technician]
    result[target, technician] = action[target]
    result[action == n+1, technician] = 0
    return result


def preceding_owner(matching, action, repair, arm, n):
    """Owner touched by this slot, or -1; reset even when the slot is KEEP."""
    if not repair or arm not in BILATERAL_ARMS:
        return torch.full((len(matching),), -1, dtype=torch.long, device=matching.device)
    claimed = (matching == action[:, None]) & ((action > 0) & (action <= n))[:, None]
    return torch.where(claimed.any(1), claimed.long().argmax(1), -1)


def calendar_table(encoder, raw):
    """Memoize identical nonlearned calendar inputs within this evaluation only.

    Integer keys are used only when reconstruction is bitwise exact. They do
    not quantize the policy's inputs. Other inputs use the literal encoder.
    """
    shape=raw.shape;flat=raw.reshape(-1,4)
    if flat.requires_grad:
        return encoder(raw)
    active=flat[:,0].round().long();duration=(flat[:,1]*12).round().long()
    age=(flat[:,2]*8).round().long();flag=flat[:,3].round().long()
    canonical=torch.stack([active.to(raw.dtype),duration.to(raw.dtype)/12,
        age.to(raw.dtype)/8,flag.to(raw.dtype)],1)
    if (not torch.equal(flat,canonical) or not ((flag>=0)&(flag<=1)&(active>=0)&(active<=1)).all()
            or (duration.abs()>1000000).any() or (age.abs()>1000000).any()):
        return encoder(raw)
    d0,a0=duration.min(),age.min()
    ds,ags=duration.max()-d0+1,age.max()-a0+1
    # Registered physical calendars have small finite ranges. Avoid integer
    # overflow for an out-of-contract synthetic input instead of aliasing it.
    if float(ds)*float(ags)*(float(active.abs().max())+1)>2**60:
        return encoder(raw)
    keys=((active*ds+duration-d0)*ags+age-a0)*2+flag
    unique,inverse=torch.unique(keys,sorted=True,return_inverse=True)
    f=unique%2;rest=unique//2
    a=rest%ags+a0;rest=rest//ags
    d=rest%ds+d0;act=rest//ds
    inputs=torch.stack([act.to(raw.dtype),d.to(raw.dtype)/12,a.to(raw.dtype)/8,f.to(raw.dtype)],1)
    return encoder(inputs)[inverse].reshape(*shape[:-1],-1)


class ExchangePolicy(nn.Module):
    is_actor = True
    joint_policy = True
    allocation_learning_mode = 'joint'

    def __init__(self, algorithm, hidden=27):
        super().__init__()
        if algorithm not in CORE_ARMS:
            raise ValueError('exchange core arm')
        self.algorithm, self.hidden = algorithm, hidden
        self.encoder = mlp(22, hidden, hidden)
        self.calendar_encoder = mlp(4, hidden, hidden)
        self.move_head = mlp(5*hidden+2, hidden, 1)
        self.control_head = mlp(2*hidden+2, hidden, 2)
        self.value_encoder = mlp(22, hidden, hidden)
        self.value_calendar_encoder = mlp(4, hidden, hidden)
        self.value_context = mlp(2*hidden+2, hidden, hidden)
        self.value_head = mlp(hidden+9, hidden, 1)

    def gradient_groups(self):
        return {label: [p for name, p in self.named_parameters()
            if name.startswith(('value_encoder.', 'value_calendar_encoder.', 'value_context.', 'value_head.')) == (label == 'critic')]
            for label in ('actor', 'critic')}

    def encode_static(self, batch, compute_values):
        m, t, e, g = (batch[key] for key in ('machines', 'technicians', 'edges', 'global_features'))
        b, n, k = e.shape[:3]
        raw = torch.cat([m[:, :, None].expand(-1, -1, k, -1),
            t[:, None].expand(-1, n, -1, -1), e, g[:, None, None].expand(-1, n, k, -1)], -1)
        base = self.encoder(raw)
        if not compute_values:
            return base, torch.zeros(b, dtype=base.dtype, device=base.device)
        calendar, planned = initial_calendar(batch)
        edge = torch.tanh(self.value_encoder(raw)+calendar_embeddings(self.value_calendar_encoder, calendar, planned))
        tech, graph = edge.mean(1), edge.mean((1, 2))
        position = torch.arange(k, dtype=base.dtype, device=base.device)[None, :, None].expand(b, -1, -1)/k
        roles = torch.cat([position, t[..., 0, None]], -1)
        context = self.value_context(torch.cat([tech, graph[:, None].expand(-1, k, -1), roles], -1)).mean(1)
        value = self.value_head(torch.cat([context, g], -1)).squeeze(-1)*value_unit(n)
        return base, value

    def literal_logits(self, batch, base, matching, technician, repair):
        b, n, k, h = base.shape
        view = intent_view(matching, technician, self.algorithm == 'independent_ppo')
        calendar, planned, intent = provisional_calendar(batch, view)
        edge = base+calendar_embeddings(self.calendar_encoder, calendar, planned)
        owners, occupied = intent.argmax(2), intent.any(2)
        rows = torch.arange(b, device=base.device)
        own = matching[:, technician]
        own_row = edge[rows, (own-1).clamp(min=0)]
        own_edge = own_row[:, technician]*(own > 0)[:, None]
        holder = edge.gather(2, owners[..., None, None].expand(-1, -1, 1, h)).squeeze(2)*occupied[..., None]
        reciprocal = own_row.gather(1, owners[..., None].expand(-1, -1, h))*occupied[..., None]*(own > 0)[:, None, None]
        graph = edge.mean((1, 2))
        phase = torch.tensor([float(repair), technician/k], dtype=base.dtype, device=base.device)[None].expand(b, -1)
        context = torch.cat([edge[:, :, technician], own_edge[:, None].expand(-1, n, -1), holder,
            reciprocal, graph[:, None].expand(-1, n, -1), phase[:, None].expand(-1, n, -1)], -1)
        moves = self.move_head(context).squeeze(-1)
        controls = self.control_head(torch.cat([own_edge, graph, phase], -1))
        return torch.cat([controls[:, :1], moves, controls[:, 1:]], -1)

    def logits(self, batch, base, matching, technician, repair, base_index=None, base_mean=None):
        # Same binary-calendar policy without a per-slot N*K*H edge tensor.
        b, k = matching.shape
        n, h = base.shape[1], base.shape[-1]
        device = base.device
        rows = torch.arange(b, device=device)
        bi = rows if base_index is None else base_index
        j = torch.as_tensor(technician, device=device).expand(b)
        phase_repair = torch.as_tensor(repair, device=device).expand(b)
        view = intent_view(matching, j, self.algorithm == 'independent_ppo')
        calendar, planned, intent = provisional_calendar(batch, view)
        raw = calendar[:, :, None, :].expand(-1, -1, 2, -1)
        flags = torch.tensor([0., 1.], device=device, dtype=base.dtype)[None,None,:,None].expand(b,n,-1,-1)
        pair = calendar_table(self.calendar_encoder,torch.cat([raw,flags],-1))
        c0, delta = pair[:,:,0], pair[:,:,1]-pair[:,:,0]
        owners, occupied = intent.argmax(2), intent.any(2)
        machine = torch.arange(n, device=device)[None,:]
        own = matching[rows,j]
        own_m = (own-1).clamp(min=0)
        own_c, own_d = c0[rows,own_m], delta[rows,own_m]
        own_edge = (base[bi,own_m,j]+own_c+planned[rows,own_m,j,None]*own_d)*(own>0)[:,None]
        candidate = base[bi[:,None],machine,j[:,None]]+c0+planned[rows[:,None],machine,j[:,None],None]*delta
        holder = (base[bi[:,None],machine,owners]+c0+planned.gather(2,owners[...,None])*delta)*occupied[...,None]
        own_flags = planned[rows,own_m].gather(1,owners)
        reciprocal = (base[bi[:,None],own_m[:,None],owners]+own_c[:,None]+own_flags[...,None]*own_d[:,None])
        reciprocal = reciprocal*occupied[...,None]*(own>0)[:,None,None]
        static_mean = base.mean((1,2)) if base_mean is None else base_mean
        graph = static_mean[bi]+c0.mean(1)+(planned.mean(2)[...,None]*delta).mean(1)
        phase = torch.stack([phase_repair.to(base.dtype),j.to(base.dtype)/k],1)
        context = torch.cat([candidate,own_edge[:,None].expand(-1,n,-1),holder,
            reciprocal,graph[:,None].expand(-1,n,-1),phase[:,None].expand(-1,n,-1)],-1)
        moves = self.move_head(context).squeeze(-1)
        controls = self.control_head(torch.cat([own_edge,graph,phase],-1))
        return torch.cat([controls[:,:1],moves,controls[:,1:]],-1)

    def replay(self, batch, recorded, base, value):
        b,n,k = base.shape[:3];slots=2*k
        matching = torch.zeros((b,k),dtype=torch.long,device=base.device)
        previous = torch.full((b,),-1,dtype=torch.long,device=base.device)
        prefixes, supports = [], []
        for slot in range(slots):
            j,repair = slot%k,slot>=k
            support = move_support(batch,matching,j,repair,self.algorithm,previous)
            action = recorded[:,slot]
            if ((action<0)|(action>n+1)).any() or not support.gather(1,action[:,None]).all():
                raise ValueError('masked exchange replay action')
            prefixes.append(matching);supports.append(support)
            previous = preceding_owner(matching,action,repair,self.algorithm,n)
            matching = apply_move(matching,action,j,repair,self.algorithm,n)
        flat = {key:batch[key][:,None].expand(-1,slots,*batch[key].shape[1:]).reshape(b*slots,*batch[key].shape[1:])
                for key in ('machines','edges')}
        index = torch.arange(b,device=base.device).repeat_interleave(slots)
        slot_index = torch.arange(slots,device=base.device).repeat(b)
        logits = self.logits(flat,base,torch.stack(prefixes,1).reshape(b*slots,k),
            slot_index%k,slot_index>=k,index,base.mean((1,2)))
        support = torch.stack(supports,1).reshape(b*slots,n+2)
        log = torch.log_softmax(masked(logits,support),-1)
        selected = log.gather(1,recorded.reshape(-1,1)).reshape(b,slots)
        entropy = -(log.exp()*log).sum(1).reshape(b,slots)
        return recorded,selected,entropy,value[:,None].expand(-1,slots)

    def trace(self, batch, recorded=None, generator=None, deterministic=False,
              compute_values=True, sequential=False, literal=False):
        base,value = self.encode_static(batch,compute_values)
        b,n,k = base.shape[:3]
        if recorded is not None and (recorded.shape!=(b,2*k) or recorded.dtype!=torch.long):
            raise ValueError('recorded exchange trace shape/type')
        if recorded is not None and not sequential and not literal:
            return self.replay(batch,recorded,base,value)
        matching = torch.zeros((b,k),dtype=torch.long,device=base.device)
        previous = torch.full((b,),-1,dtype=torch.long,device=base.device)
        actions,logs,entropies = [],[],[]
        base_mean = base.mean((1,2))
        for slot in range(2*k):
            j,repair=slot%k,slot>=k
            logits = (self.literal_logits(batch,base,matching,j,repair) if literal else
                      self.logits(batch,base,matching,j,repair,base_mean=base_mean))
            support = move_support(batch,matching,j,repair,self.algorithm,previous)
            log = torch.log_softmax(masked(logits,support),-1)
            action = (recorded[:,slot] if recorded is not None else log.argmax(1) if deterministic else
                      torch.multinomial(log.exp(),1,generator=generator).squeeze(1))
            if ((action<0)|(action>n+1)).any() or not support.gather(1,action[:,None]).all():
                raise ValueError('masked exchange trace action')
            actions.append(action);logs.append(log.gather(1,action[:,None]).squeeze(1))
            entropies.append(-(log.exp()*log).sum(1))
            previous=preceding_owner(matching,action,repair,self.algorithm,n)
            matching=apply_move(matching,action,j,repair,self.algorithm,n)
        return torch.stack(actions,1),torch.stack(logs,1),torch.stack(entropies,1),value[:,None].expand(-1,2*k)

    def evaluate(self,batch,actions):
        _,log,entropy,value=self.trace(batch,recorded=actions)
        return log,entropy,value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if not deterministic and generator is None:
            raise ValueError('explicit action RNG')
        action, log, _, value = self.trace(batch, generator=generator, deterministic=deterministic)
        return action, log, value

    @torch.no_grad()
    def select(self, observation):
        return self.trace(tensors([observation]), deterministic=True, compute_values=False)[0][0].numpy().copy()

    def environment_action(self, action):
        return action.numpy().copy()

    def return_diagnostics(self, returns, old_values):
        unit = value_unit(old_values.shape[1])  # 2K=N in the registered settings.
        return dict(physical_value_unit=unit, return_mean=float(returns.mean()),
            return_std=float(returns.std(unbiased=False)), value_mean=float(old_values.mean()),
            normalized_return_mean=float(returns.mean()/unit), normalized_return_std=float(returns.std(unbiased=False)/unit),
            normalized_value_mean=float(old_values.mean()/unit), normalized_value_std=float(old_values.std(unbiased=False)/unit),
            normalized_prefix_baseline_spread=float(old_values.std(1, unbiased=False).mean()/unit))
