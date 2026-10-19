from pathlib import Path
import csv,statistics,json
from collections import defaultdict
from ht_pdm_fjsp import passive_technician_heuristic_assignment as e
OUT=Path('/Users/bachng/Coding/Reinforcement Learning/marl/reports/ra_qmix_heuristic_assignment_analysis_20261005T025516Z')
rows=[]
for n in range(2,6):
    configs=e.cells('full',n)
    for c in ('train_a','train_b'):
        for s in e.settings('full')['evaluation_seeds']:
            rows.append(dict(machines=n,cell=c,algorithm='heuristic_reference',**e.rollout(configs[c],s)))
e.base._write_csv(OUT/'exploratory_training_teacher_episodes.csv',rows)
means=[]
for n in range(2,6):
    for c in ('train_a','train_b'):
        rr=[r for r in rows if r['machines']==n and r['cell']==c]
        means.append(dict(machines=n,cell=c,objective=statistics.fmean(r['objective'] for r in rr),collisions=statistics.fmean(r['collisions'] for r in rr),failures=statistics.fmean(r['failures'] for r in rr),waiting=statistics.fmean(r['waiting'] for r in rr)))
e.base._write_csv(OUT/'exploratory_training_teacher_means.csv',means)
print(json.dumps(means,indent=2))
