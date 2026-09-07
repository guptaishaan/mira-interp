"""Independent auditor/producer agreement on artificial complete dose cohorts."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
import numpy as np
import pytest
import evaluate_rollouts
from audit_generated_measurements import equal,independent_summary,ordering_independent
from test_generated_evaluation import fixture_records,config


@pytest.mark.parametrize('slopes',[None,{('a',1):0.,('a',2):1.,('b',1):-1.,('b',2):-1.},
                                   {('a',1):0.,('a',2):0.,('b',1):0.,('b',2):0.}])
def test_complete_independent_summary_agrees_on_positive_mixed_and_constant(slopes):
    records,prediction=fixture_records(slopes,ego_shift=2.)
    scale=np.arange(1.,13.)
    actual=independent_summary(records,prediction,scale,config())
    expected=evaluate_rollouts.summarize(records,prediction,scale,config())
    for key in ('per_pair','summary'):
        for a,b in zip(actual[key],expected[key]):equal(a,b)
    assert actual['matches']==expected['matches']==2


def test_consistent_clipped_curves_match_all_metrics():
    records,prediction=fixture_records();lookup={r['record_id']:i for i,r in enumerate(records)}
    for i,row in enumerate(records):
        effective=float(np.clip(row['dose'],-100,400));row['height']['effective_dose']=effective
        row['height']['target_clipped']=effective!=row['dose']
        if row['dose']:
            base=prediction[lookup[row['baseline_record_id']]]
            prediction[i]=base+(prediction[i]-base)*(effective/row['dose'])
    actual=independent_summary(records,prediction,np.ones(12),config())
    expected=evaluate_rollouts.summarize(records,prediction,np.ones(12),config())
    for key in ('per_pair','summary'):
        for a,b in zip(actual[key],expected[key]):equal(a,b)


def test_rank_ties_constant_and_inconsistent_duplicate_handled_independently():
    x=np.array([-600,-300,0,0,0.])
    for y in (np.array([-2,-1,0,-10,10.]),np.zeros(5),2*x):
        equal(ordering_independent(x,y),evaluate_rollouts.ordering(x,y))
