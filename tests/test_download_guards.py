"""A stale local file must not bypass the download disk-reserve check."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).parents[1] / "scripts/prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


def test_cached_credit_requires_publisher_content_hash(tmp_path):
    path = tmp_path / "data"
    good = b"complete"
    meta = SimpleNamespace(size=len(good), lfs=SimpleNamespace(sha256=hashlib.sha256(good).hexdigest()))
    assert prepare_data.verified_cached_bytes(path, meta) == 0
    path.write_bytes(b"incomplete")
    assert prepare_data.verified_cached_bytes(path, meta) == 0
    path.write_bytes(b"corrupt!")
    assert prepare_data.verified_cached_bytes(path, meta) == 0
    path.write_bytes(good)
    assert prepare_data.verified_cached_bytes(path, meta) == len(good)
    meta = SimpleNamespace(size=len(good), lfs=None,
                           blob_id=hashlib.sha1(b"blob 8\0" + good).hexdigest())
    assert prepare_data.verified_cached_bytes(path, meta) == len(good)
    path.write_bytes(b"corrupt!")
    assert prepare_data.verified_cached_bytes(path, meta) == 0
