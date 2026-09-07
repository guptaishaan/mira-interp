"""Publication excludes observed context and sizes real deterministic tar parts."""
import hashlib
import importlib.util
from pathlib import Path
import tarfile
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('generated_package',Path(__file__).resolve().parents[1]/'scripts/package_generated_outputs.py')
pkg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pkg)


def test_exact_future_slices_and_no_extra_arrays():
    source={key:np.broadcast_to(np.array(0.,dtype=np.float32),shape) for key,shape in pkg.SHAPES.items()}
    source['frames']=np.broadcast_to(np.arange(24,dtype=np.float32).reshape(1,24,1,1,1)/24,pkg.SHAPES['frames'])
    source['latents']=np.broadcast_to(np.arange(12,dtype=np.float32).reshape(1,12,1,1,1),pkg.SHAPES['latents'])
    future=pkg.public_arrays(source)
    assert set(future)==set(pkg.PUBLIC_SHAPES)
    assert {k:v.shape for k,v in future.items()}==pkg.PUBLIC_SHAPES
    np.testing.assert_array_equal(future['frames'][0,:,0,0,0],np.arange(16,24,dtype=np.float32)/24)
    np.testing.assert_array_equal(future['latents'][0,:,0,0,0],np.arange(8,12,dtype=np.float32))
    assert future['native_edit_tile'] is source['native_edit_tile']
    with pytest.raises(ValueError):pkg.public_arrays({**source,'actions':np.zeros((4,24,9),np.uint8)})
    with pytest.raises(ValueError):pkg.public_arrays({**source,'targets':np.zeros((4,24,30))})


def test_partition_uses_exact_tar_padding_and_deterministic_readback(tmp_path):
    entries=[]
    for i in range(3):
        path=tmp_path/f'{i}.bin';path.write_bytes(bytes([i])*16000)
        entries.append({'member':f'metadata/{i}.bin','staged_path':str(path),'bytes':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kind':'metadata_json'})
    groups=pkg.partition(entries,'pilot',{},40960)
    assert len(groups)==2 and sum(map(len,groups))==3
    for i,group in enumerate(groups,1):
        manifest=pkg.part_manifest(group,i,'pilot',{})
        a,b=tmp_path/f'a{i}.tar',tmp_path/f'b{i}.tar'
        pkg.write_part(a,group,manifest);pkg.write_part(b,group,manifest)
        assert a.read_bytes()==b.read_bytes()
        assert a.stat().st_size==pkg.part_size(group,i,'pilot',{})<=40960
        pkg.verify_part(a,group,manifest)
    first=tmp_path/'a1.tar'
    with tarfile.open(first) as tar:offset=tar.getmember(groups[0][0]['member']).offset_data
    with first.open('r+b') as stream:stream.seek(offset);stream.write(b'x')
    with pytest.raises(ValueError):pkg.verify_part(first,groups[0],pkg.part_manifest(groups[0],1,'pilot',{}))


@pytest.mark.parametrize('name',['../escape','/absolute','x/../../escape','x\\escape'])
def test_unsafe_archive_member_paths_rejected(name):
    with pytest.raises(ValueError):pkg.safe_name(name)
