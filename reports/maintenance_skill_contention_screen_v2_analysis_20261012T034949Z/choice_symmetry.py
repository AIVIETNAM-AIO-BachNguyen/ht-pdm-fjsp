"""Post hoc diagnostic of every contested max-cardinality machine subset."""
from collections import Counter
import json
from pathlib import Path
import sys

from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
sys.path.insert(0, str(CODE / 'src'))
from ht_pdm_fjsp import maintenance_skill_contention_screen_v2 as s
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

analysis = json.loads((HERE / 'analysis.json').read_text())
run = Path(analysis['run'])
cfg = json.loads((run / 'resolved_config.json').read_text())['settings']
c = s.decode_config(analysis['selected_setting'])
counts = Counter()
examples = []
for seed in tqdm(cfg['confirmation_seeds'], desc='Check all contested choice subsets', unit='seed'):
    env = WaitingEnv(c); env.reset(role_seed(seed, f'{s.VERSION}:physical'))
    for tick in range(c.horizon):
        state = env.state
        space = s.action_space(c, state)
        action = s.choose(c, state, 'deadline_matching', space)
        if tick < cfg['horizon']:
            metrics = s.state_metrics(c, state, action, space)
            if metrics['contested_choice_rate']:
                counts['joint_ticks'] += 1
                maximum = max(map(len, space[2]))
                subsets = [x for x in s.base.physical_choices(c, space[2]) if len(x) == maximum]
                heterogeneous = []
                for subset in subsets:
                    signatures = {(state.ages[m], state.failed[m], state.pending_wait[m],
                        c.service_time[m], c.restoration[m]) for m in subset}
                    counts['choice_subsets'] += 1
                    counts['heterogeneous_choice_subsets'] += len(signatures) > 1
                    if len(signatures) > 1: heterogeneous.append(subset)
                counts['joint_ticks_with_heterogeneous_choice'] += bool(heterogeneous)
                if len(examples) < 3:
                    examples.append(dict(seed=seed, tick=tick, ages=state.ages, failed=state.failed,
                        pending_wait=state.pending_wait, possible_choice_subsets=subsets,
                        heterogeneous_subsets=heterogeneous))
        env.step(action)
counts.setdefault('joint_ticks_with_heterogeneous_choice', 0)
counts.setdefault('heterogeneous_choice_subsets', 0)
result = dict(scope='posthoc; 128 winner confirmation seeds, one rule because both paths are identical',
    counts=dict(counts), examples=examples,
    limitation='Physical assignment alternatives do not demonstrate different expected economic value. Symmetric machines can exchange fast and slow workers without breaking model symmetry; fixed sampled future shock tapes need not give identical realized costs.')
(HERE / 'choice_symmetry.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
