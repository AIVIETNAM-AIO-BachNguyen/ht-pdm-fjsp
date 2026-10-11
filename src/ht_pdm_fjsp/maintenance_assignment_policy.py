"""Parameter-matched pair/resource scoring and PM-only conditional decoding."""
import itertools
import numpy as np
import torch
from torch.distributions import Categorical
from ht_pdm_fjsp.maintenance_autoregressive_policy import AutoregressivePolicy, CONTEXT, STOP
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE

ARMS = ("additive_ppo", "interaction_ppo")
DECODERS = ("greedy", "assignment_mass")
TOKEN_TOLERANCE = 1e-5
MASS_TOLERANCE = 1e-6


def tie_argmax(logits):
    """Lowest valid token within the common deterministic logit tolerance."""
    if logits.ndim != 2 or not torch.isfinite(logits.max(1).values).all():
        raise ValueError("finite row maxima required")
    return (logits >= logits.max(1, keepdim=True).values-TOKEN_TOLERANCE).long().argmax(1)


def controller(arm, decoder):
    if arm not in ARMS or decoder not in DECODERS:
        raise ValueError("unknown scorer/decoder")
    return f"{arm}:{decoder}"


class AssignmentPolicy(AutoregressivePolicy):
    def __init__(self, arm=ARMS[0], hidden=32, horizon=48, tail=12):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(CONTEXT, hidden)
        self.experiment_arm = arm
        self.decoder = "greedy"
        self.learned_horizon, self.tail_ticks = horizon, tail

    def token_distribution(self, batch, encoded, live, count):
        if self.experiment_arm == ARMS[0]:
            return super().token_distribution(batch, encoded, live, count)
        rem_m, rem_t = self.resource_context(encoded, live, batch["pair_mask"])
        b, h = len(live), self.hidden
        expand = lambda x: x[:, None].expand(-1, 9, -1)
        pair = torch.cat([encoded["pair"].reshape(b, 8, h), torch.zeros(b, 1, h)], 1)
        machines = encoded["machine"][:, :, None].expand(-1, 4, 2, -1).reshape(b, 8, h)
        machines = torch.cat([machines, torch.zeros(b, 1, h)], 1)
        # Candidate-dependent resource interaction, no new weights or IDs.
        machines = machines + pair*expand((encoded["g"]+rem_m+rem_t)/3)
        context = torch.cat([expand(encoded["g"]), expand(encoded["machine"].mean(1)),
            expand(encoded["tech"].mean(1)), pair, machines, expand(rem_m), expand(rem_t),
            (count.float()/4)[:, None, None].expand(-1, 9, -1),
            (batch["technicians"][:, :, 0].sum(1)/2)[:, None, None].expand(-1, 9, -1)], -1)
        mask = torch.cat([live.reshape(b, 8), torch.ones(b, 1, dtype=torch.bool)], 1)
        return Categorical(logits=self.actor(context).squeeze(-1).masked_fill(~mask, -torch.inf)), mask

    @torch.no_grad()
    def greedy_tokens(self, batch):
        encoded = self.encode(batch)
        live = batch["pair_mask"].clone()
        b = len(live)
        count = torch.zeros(b, dtype=torch.long)
        active = torch.ones(b, dtype=torch.bool)
        tokens = []
        for _ in range(3):
            distribution, _ = self.token_distribution(batch, encoded, live, count)
            token = tie_argmax(distribution.logits)
            tokens.append(torch.where(active, token, -1))
            chosen = active & (token < STOP)
            rows = torch.arange(b)[chosen]
            m, j = token[chosen]//2, token[chosen]%2
            live[rows, m, :] = False
            live[rows, :, j] = False
            count += chosen.long()
            active = chosen
        if active.any():
            raise RuntimeError("missing STOP")
        return torch.stack(tokens, 1)

    def trace(self, batch, recorded=None, generator=None, deterministic=False):
        if deterministic and recorded is None:
            recorded = self.greedy_tokens(batch)
        return super().trace(batch, recorded, generator, False)

    @torch.no_grad()
    def conditional_decode(self, batch, greedy):
        """Exact unordered mass on fixed machines/fixed CM; original wins ties."""
        candidates, owner, traces = [], [], []
        selected_sets = []
        for row, tokens in enumerate(greedy.tolist()):
            original = tuple(sorted((x//2, x%2) for x in tokens if 0 <= x < STOP))
            machines = {m for m, _ in original}
            cm = {edge for edge in original if batch["machines"][row, edge[0], 1] > .5}
            allowed = [a for a in CATALOGUE if {m for m, _ in a} == machines
                       and cm.issubset(a) and all(batch["pair_mask"][row, m, j] for m, j in a)]
            allowed = [original, *[a for a in allowed if a != original]]
            selected_sets.append(allowed)
            for local, action in enumerate(allowed):
                for order in itertools.permutations(action):
                    candidates.append((row, local));owner.append(row)
                    traces.append([2*m+j for m, j in order]+[STOP]+[-1]*(2-len(action)))
        ids = torch.tensor(owner)
        sub = {k: v[ids] for k, v in batch.items()}
        lp, _, _ = self.evaluate(sub, torch.tensor(traces, dtype=torch.long))
        masses = [np.zeros(len(actions)) for actions in selected_sets]
        for (row, local), value in zip(candidates, lp.exp().tolist(), strict=True):
            masses[row][local] += value
        result, audit = [], []
        for row, (allowed, values) in enumerate(zip(selected_sets, masses, strict=True)):
            best = next(i for i, value in enumerate(values) if value >= values.max()-MASS_TOLERANCE)
            action = allowed[best]
            result.append([2*m+j for m, j in action]+[STOP]+[-1]*(2-len(action)))
            audit.append(dict(original=allowed[0], selected=action, candidates=allowed,
                masses=values.tolist(), changed=best != 0,
                probability_semantics="sum ordered trace masses; conditioned by fixed machines/CM"))
        return torch.tensor(result, dtype=torch.long), audit

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False):
        tokens, prob, value = super().sample(batch, generator, deterministic)
        self.last_decode_audit = None
        if deterministic and self.decoder == "assignment_mass":
            tokens, self.last_decode_audit = self.conditional_decode(batch, tokens)
            prob, _, value = self.evaluate(batch, tokens)
        return tokens, prob, value
