"""Synthetic integrity checks for the independent auditor; no rollout outcomes."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('generation_audit', Path(__file__).resolve().parents[1]/'scripts/audit_rollout_generation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def trace_fixture():
    schedule = [0., .1, .2, .3, .4, .45, .48, .5, .6, .8, 1.]
    registration = {'schedule_values_fp32':schedule, 'schedule_values_bf16':schedule}
    trace = []
    for index in range(45):
        if index == 0:
            phase, latent, step, tau, time, cache, ret, target = 'context_cache_initialization',-1,-1,1.,8,False,True,False
        else:
            latent, position = divmod(index-1,11)
            cache_update = position == 10
            phase = 'generated_cache_update' if cache_update else 'denoising'
            step, tau = (9,1.) if cache_update else (position,schedule[position])
            time, cache, ret, target = 1,True,cache_update,latent == 0 and position == 8
        trace.append(dict(call_index=index,phase=phase,generated_latent_index=latent,step_index=step,tau=tau,
            time_tokens=time,cache_present=cache,return_kv=ret,target_call=target,latent_dtype='torch.float32',
            tau_dtype='torch.float32',nominal_schedule_tau=tau,latent_input_sha256='a'*64,action_features_sha256='b'*64))
    return trace,registration


def test_trace_checks_actual_steps_cache_and_unique_edit():
    trace,registration = trace_fixture()
    assert audit.validate_trace(trace,registration)
    corruptions = [(9,'target_call',False),(20,'target_call',True),(11,'return_kv',False),
                   (1,'generated_latent_index',1),(9,'tau',.5),(0,'time_tokens',16),
                   (30,'action_features_sha256','bad'),(0,'cache_present',True)]
    for index,key,value in corruptions:
        bad = copy.deepcopy(trace);bad[index][key] = value
        with pytest.raises(ValueError):
            audit.validate_trace(bad,registration)
    with pytest.raises(ValueError):
        audit.validate_trace(trace[:-1],registration)


def test_independent_descriptor_lift_is_right_inverse():
    q = np.zeros((2048,128),np.float32);q[:128] = np.eye(128)
    rng = np.random.default_rng(2)
    delta = rng.normal(size=(1,1536))
    lifted = audit.lift(delta,q)
    np.testing.assert_allclose(audit.descriptor(lifted,q),delta,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(lifted)**2,12*np.linalg.norm(delta)**2,rtol=1e-14)


def test_numerical_comparison_rejects_nan_and_broadcast():
    with pytest.raises(ValueError):audit.close([1],[np.nan],'bad')
    with pytest.raises(ValueError):audit.close(np.ones((1,2)),np.ones(2),'bad')
    with pytest.raises(ValueError):audit.close([1],[2],'bad')
    audit.close([1],[1],'good')


def test_reserved_pilot_support_only_dose_rule():
    assert audit.pilot_dose_for_base(-999) == 300.
    assert audit.pilot_dose_for_base(600) == 300.
    assert audit.pilot_dose_for_base(audit.SUPPORT[1]-1) == 300.
    assert audit.pilot_dose_for_base(audit.SUPPORT[1]) == -300.
    assert audit.pilot_dose_for_base(audit.SUPPORT[1]+999) == -300.
    with pytest.raises(ValueError):audit.pilot_dose_for_base(np.nan)
