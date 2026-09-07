"""No GPU or real generated videos: endpoint, units, and paired-control contracts."""
import copy
import numpy as np
import pytest
from mira_interp.rollout_evaluator import END_FRAMES, absolute_ball, paired_changes, validate_pairs, validate_generated_video, window_indices


def record(key, baseline='zero', dose=0., seed=1):
    return dict(record_id=key,baseline_record_id=baseline,dose=dose,seed=seed,match_id='m',clip_id='c',split='selection',
                intervention_type='baseline' if dose==0 else 'linear',generated_artifact_path='/private/video.npz',generated_sha256='0'*64)


def test_windows_end_on_each_generated_tubelet_without_later_frames():
    indices = window_indices()
    assert indices.shape == (4,16)
    np.testing.assert_array_equal(indices[:,0],[2,4,6,8]);np.testing.assert_array_equal(indices[:,-1],END_FRAMES)
    assert np.all(np.diff(indices,axis=1)==1)
    source = np.arange(24)
    np.testing.assert_array_equal(source[indices][:,-1],[17,19,21,23])


def test_absolute_ball_retains_covariance_via_exact_coordinate_sum():
    values = np.zeros((4,4,12));values[...,:6]=np.arange(6)+100;values[...,6:]=np.arange(6)*2-100
    np.testing.assert_array_equal(absolute_ball(values),np.broadcast_to(np.arange(6)*3,(4,4,6)))
    with pytest.raises(ValueError,match='finite'):absolute_ball(np.full((4,4,12),np.nan))


def test_paired_changes_use_explicit_same_seed_baseline_and_all_views_times():
    rows = [record('zero'),record('plus',dose=300)]
    base = np.arange(4*4*12).reshape(4,4,12).astype(float);edit=base.copy();edit[...,2]+=2;edit[...,8]+=3
    result = paired_changes(rows,{'zero':base,'plus':edit})
    np.testing.assert_array_equal(result['zero']['role_delta'],0)
    np.testing.assert_array_equal(result['plus']['absolute_ball_delta'][...,2],5)
    np.testing.assert_array_equal(result['plus']['role_delta'][...,0],0)
    wrong=copy.deepcopy(rows);wrong[1]['seed']=2
    with pytest.raises(ValueError,match='identical source/seed'):validate_pairs(wrong)
    with pytest.raises(ValueError,match='missing'):validate_pairs([rows[1]])
    wrong=copy.deepcopy(rows);wrong[0]['dose']=1
    with pytest.raises(ValueError,match='zero dose'):validate_pairs(wrong)


def test_source_pixel_units_and_native_video_contract_are_not_silently_guessed():
    shape=(4,24,3,288,512)
    valid=np.broadcast_to(np.asarray(.5,dtype=np.float32),shape)
    assert validate_generated_video(valid) is valid
    with pytest.raises(ValueError,match='0..1'):
        validate_generated_video(np.broadcast_to(np.asarray(255,dtype=np.float32),shape))
    with pytest.raises(ValueError,match='float32'):
        validate_generated_video(np.zeros((4,16,3,2,2),dtype=np.float32))
