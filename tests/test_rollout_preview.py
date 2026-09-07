"""Publication boundary and view/time identity checks, without research data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/export_rollout_preview.py"
    spec = importlib.util.spec_from_file_location("rollout_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_future_export_excludes_context_and_preserves_view_time_identity():
    module = load_script()
    frames = np.full((4, 24, 3, 288, 512), np.nan, dtype=np.float32)
    for view in range(4):
        for time in range(8):
            frames[view, 16 + time] = np.float32((view * 8 + time) / 32)
    future = module.future_uint8(frames)
    assert future.shape == (4, 8, 288, 512, 3)
    for view in range(4):
        for time in range(8):
            assert np.all(future[view, time] == np.rint(np.float32((view * 8 + time) / 32) * 255))
    grid = module.four_view_video(future, "fixture")
    for view in range(4):
        y, x = 64 + (view // 2) * 288 + 200, (view % 2) * 512 + 400
        np.testing.assert_array_equal(grid[:, y, x], future[view, :, 200, 400])


def test_future_export_rejects_wrong_time_axis_and_nonfinite_future():
    module = load_script()
    with pytest.raises(ValueError, match="Expected saved"):
        module.future_uint8(np.zeros((4, 8, 3, 288, 512), dtype=np.float32))
    frames = np.zeros((4, 24, 3, 288, 512), dtype=np.float32)
    frames[2, 20, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Generated pixels"):
        module.future_uint8(frames)
