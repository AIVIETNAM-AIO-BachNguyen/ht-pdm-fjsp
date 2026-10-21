import json
from pathlib import Path

from research_result.probe_v14_training_credit import SOURCE,SOURCE_AUDIT,TRAIN,WAIT,summarize


def test_audited_training_source_and_wait_action():
    source=json.loads((SOURCE/'manifest.json').read_text())
    report=json.loads(SOURCE_AUDIT.read_text())
    assert source['status']=='COMPLETED' and report['status']=='PASS'
    assert Path(report['run']).resolve()==SOURCE
    assert WAIT==0 and len(TRAIN)==5


def test_summary_separates_q_preference_and_actor_choice():
    rows=[]
    for case in ('interchangeable_scale1','reciprocal_scale2'):
        rows.extend([
            dict(case=case,train_seed=TRAIN[0],early=True,q_margin=10.,selected_action=1,selected_regret=10.),
            dict(case=case,train_seed=TRAIN[1],early=True,q_margin=-10.,selected_action=WAIT,selected_regret=10.),
            dict(case=case,train_seed=TRAIN[2],early=False,q_margin=0.,selected_action=WAIT,selected_regret=0.),
        ])
    summary=summarize(rows)
    for case in ('interchangeable_scale1','reciprocal_scale2'):
        early=summary[case]['early'];all_rows=summary[case]['all']
        assert early['count']==2 and all_rows['count']==3
        assert early['q_wait_better_5_fraction']==.5
        assert early['selected_service_given_wait_q5']==1.
        assert early['selected_wait_given_service_q5']==1.
        assert early['selected_regret_mean']==10.
