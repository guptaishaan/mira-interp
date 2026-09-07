"""No GPU: reject content changes while permitting named storage metadata changes."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('storage_parity',Path(__file__).resolve().parents[1]/'scripts/audit_rollout_storage_parity.py')
parity=importlib.util.module_from_spec(spec);spec.loader.exec_module(parity)


def test_bitwise_content_includes_signed_zero_dtype_shape_and_layout():
    assert not parity.bitwise_array_equal(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert not parity.bitwise_array_equal(np.ones(3,np.float32),np.ones(3,np.float64))
    assert not parity.bitwise_array_equal(np.ones((1,3)),np.ones(3))
    a=np.arange(12,dtype=np.float32).reshape(3,4)[:,::2]
    assert parity.bitwise_array_equal(a,a.copy())
    b=a.copy();b[0,0]+=1
    assert not parity.bitwise_array_equal(a,b)


def test_only_explicit_storage_timing_metadata_can_differ():
    old={'generated_bytes':1,'elapsed_seconds':1.,'dose':300.,'action_batch_sha256':'a','world_model_calls':[1,2]}
    new={**old,'generated_bytes':2,'elapsed_seconds':9.}
    parity.equal_except(old,new,parity.ROW_IGNORED,'unexpected')
    for key,value in [('dose',-300.),('action_batch_sha256','b'),('world_model_calls',[1,3]),('new_unreviewed_field',True)]:
        with pytest.raises(ValueError):parity.equal_except(old,{**new,key:value},parity.ROW_IGNORED,'changed')
