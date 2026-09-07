"""Storage format changes preserve array bits, layouts and safe member names."""
import io
import zipfile
import numpy as np
import pytest
from mira_interp.array_storage import write_npz


def test_level1_roundtrip_noncontiguous_arrays_and_deterministic_bytes():
    arrays = {'frames': np.arange(120,dtype=np.float32).reshape(4,5,6)[:,::2,::-1],
              'fortran': np.asfortranarray(np.arange(12,dtype=np.float64).reshape(3,4)),
              'metadata': np.array(['view0','view1'])}
    original = {k:v.copy() for k,v in arrays.items()}
    first,second = io.BytesIO(),io.BytesIO()
    write_npz(first,arrays);write_npz(second,arrays)
    assert first.getvalue() == second.getvalue()
    first.seek(0)
    with zipfile.ZipFile(first) as z:
        assert all(m.compress_type == zipfile.ZIP_DEFLATED for m in z.infolist())
    first.seek(0)
    with np.load(first,allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key,value in arrays.items():
            assert saved[key].dtype == value.dtype and saved[key].shape == value.shape
            assert saved[key].tobytes() == value.tobytes()
            assert np.array_equal(value,original[key])


@pytest.mark.parametrize('arrays',[{}, {'../escape': np.zeros(1)}, {'a/b':np.zeros(1)},
                                   {'123':np.zeros(1)}, {'x':np.array([object()],dtype=object)}])
def test_invalid_names_empty_and_pickle_payload_rejected(arrays):
    with pytest.raises(ValueError):write_npz(io.BytesIO(),arrays)


def test_only_registered_compression_level_is_accepted():
    with pytest.raises(ValueError):write_npz(io.BytesIO(),{'x':np.zeros(1)},compresslevel=6)
