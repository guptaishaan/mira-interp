"""Synthetic metric controls only: no real clips, fitting, model loading or GPUs."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_rollouts as evaluation
from mira_interp.rollout_evaluator import absolute_ball, window_indices, validate_pairs


def fixture_records(pair_slopes=None, ego_shift=0.):
    pair_slopes=pair_slopes or {('a',1):1.,('a',2):3.,('b',1):5.,('b',2):7.}
    records=[];predictions=[]
    for (match,seed),slope in pair_slopes.items():
        baseline=f'{match}_{seed}_baseline'
        grid=[('baseline',0.)]+[(path,dose) for path in evaluation.PATHS for dose in evaluation.DOSES if dose]
        for path,dose in grid:
            identifier=baseline if path=='baseline' else f'{match}_{seed}_{path}_{dose}'
            records.append({'record_id':identifier,'match_id':match,'clip_id':match+'_clip','split':'selection','seed':seed,
                'intervention_type':path,'dose':dose,'baseline_record_id':baseline,'generated_artifact_path':'/synthetic/'+identifier,
                'generated_sha256':'a'*64,'height':{'effective_dose':dose,'target_clipped':False},
                'decoded_context_bitwise_equal_to_baseline':True})
            values=np.zeros((4,4,12),float)
            for view in range(4):
                for time in range(4):
                    values[view,time,2]=100+100*view+10*time+seed
                    values[view,time,8]=50+20*view+time
                    values[view,time,8]+=dose*slope*(view+1)*(time+1)
                    if dose:values[view,time,2]+=ego_shift
            predictions.append(values)
    validate_pairs(records)
    return records,np.asarray(predictions)


def config():return {'bootstrap_seed':2026090703,'bootstrap_draws':1000,'seconds_after_context':[.1,.2,.3,.4]}


def test_absolute_ball_adds_corresponding_position_and_velocity():
    values=np.arange(2*4*4*12).reshape(2,4,4,12)
    expected=np.stack([values[...,i]+values[...,i+6] for i in range(6)],axis=-1)
    assert np.array_equal(absolute_ball(values),expected)
    assert np.array_equal(window_indices(),np.array([np.arange(a-15,a+1) for a in (17,19,21,23)]))


@pytest.mark.parametrize('scale,rank,positive,success',[(2.,1.,1.,True),(-2.,-1.,0.,False),(0.,None,0.,False)])
def test_positive_negative_and_constant(scale,rank,positive,success):
    x=np.asarray(evaluation.DOSES)
    result=evaluation.ordering(x,scale*x+7)
    if rank is None:assert result['spearman'] is None
    else:assert result['spearman']==pytest.approx(rank)
    assert result['slope']==pytest.approx(scale)
    assert result['positive_adjacent_fraction']==pytest.approx(positive)
    assert result['nondecreasing_with_nonzero_span'] is success


def test_consistent_clipped_doses_use_actual_offsets():
    x=np.array([-100,-100,0,300,400.])
    result=evaluation.ordering(x,3*x+2)
    assert result['distinct_effective_doses']==4
    assert result['slope']==pytest.approx(3.)
    assert result['spearman']==pytest.approx(1.)
    assert result['nondecreasing_with_nonzero_span']


def test_all_clipped_to_same_value_cannot_count_as_success():
    result=evaluation.ordering(np.zeros(5),np.ones(5))
    assert result['distinct_effective_doses']==1
    assert result['spearman'] is None and result['slope'] is None
    assert result['nondecreasing_with_nonzero_span'] is False


def test_contradictory_duplicate_effective_doses_are_not_monotonic_success():
    result=evaluation.ordering(np.array([-600,-300,0,0,0]),np.array([-2,-1,0,-10,10]))
    assert result['nondecreasing_with_nonzero_span'] is False


def test_four_views_times_paired_seeds_and_whole_matches():
    records,predictions=fixture_records()
    report=evaluation.summarize(records,predictions,np.ones(12),config())
    assert len(report['per_pair'])==20 and len(report['summary'])==20
    for row in report['summary']:
        t=row['time_index']+1
        expected=np.array([2400.,7200.])*t
        assert row['view0_extreme_contrast']['mean']==pytest.approx(expected.mean())
        assert row['allviews_extreme_contrast']['mean']==pytest.approx(expected.mean()*2.5)
        draws=np.random.default_rng(config()['bootstrap_seed']).integers(2,size=(1000,2))
        assert np.allclose(row['view0_extreme_contrast']['match_bootstrap_ci95'],np.quantile(expected[draws].mean(1),[.025,.975]))
        assert row['positive_contrast_matches']==2
        assert row['nondecreasing_nonzero_pair_fraction']==1
        assert row['view0_nontarget_role_rms']['mean']==0
        assert np.allclose(row['view0_mean_height_by_requested_dose'],np.array(evaluation.DOSES)*4*t)


def test_collateral_excludes_only_relative_ball_z():
    records,predictions=fixture_records(ego_shift=10.)
    scale=np.ones(12);scale[2]=2
    report=evaluation.summarize(records,predictions,scale,config())
    for row in report['summary']:
        assert row['view0_nontarget_role_rms']['mean']==pytest.approx(5/np.sqrt(11))
    # Adding arbitrary target-role changes cannot alter collateral.
    changed=predictions.copy();changed[:,:,:,8]*=1000
    second=evaluation.summarize(records,changed,scale,config())
    for first,last in zip(report['summary'],second['summary']):
        assert first['view0_nontarget_role_rms']==last['view0_nontarget_role_rms']


def test_undefined_rank_seed_does_not_change_match_weight():
    records,predictions=fixture_records({('a',1):0.,('a',2):1.,('b',1):-1.,('b',2):-1.})
    report=evaluation.summarize(records,predictions,np.ones(12),config())
    for row in report['summary']:
        assert row['mean_match_spearman']==pytest.approx(0.)
        assert row['eligible_rank_matches']==2 and row['undefined_rank_matches']==0
        assert row['undefined_rank_pairs']==1


def test_constant_all_pairs_is_not_a_positive_response():
    records,predictions=fixture_records({('a',1):0.,('a',2):0.,('b',1):0.,('b',2):0.})
    report=evaluation.summarize(records,predictions,np.ones(12),config())
    for row in report['summary']:
        assert row['nondecreasing_nonzero_pair_fraction']==0
        assert row['positive_contrast_matches']==0
        assert row['undefined_rank_pairs']==4
        assert row['view0_extreme_contrast']['mean']==0


def test_paired_path_minus_controls_uses_same_match_seeds():
    records,predictions=fixture_records()
    lookup={r['record_id']:i for i,r in enumerate(records)}
    for i,row in enumerate(records):
        factor={'random_norm':.2,'wrong_variable_ball_x':-.1}.get(row['intervention_type'])
        if factor is not None:
            base=predictions[lookup[row['baseline_record_id']]]
            predictions[i]=base+factor*(predictions[i]-base)
    result=evaluation.summarize(records,predictions,np.ones(12),config())
    for row in result['summary']:
        t=row['time_index']+1
        factor={'random_norm':.2,'wrong_variable_ball_x':-.1}.get(row['path'],1.)
        for control,scale in [('random_norm',.2),('wrong_variable_ball_x',-.1)]:
            expected=np.array([2400.,7200.])*t*(factor-scale)
            value=row['view0_contrast_minus_'+control]
            assert value['mean']==pytest.approx(expected.mean())
            draws=np.random.default_rng(config()['bootstrap_seed']).integers(2,size=(1000,2))
            assert np.allclose(value['match_bootstrap_ci95'],np.quantile(expected[draws].mean(1),[.025,.975]))
