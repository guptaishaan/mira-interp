"""Synthetic unit fixtures only; these tests contain no Rocket Science results."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mira_interp.data import (
    PhysicsNormalizer, audit_future_actions, make_split_manifest, match_split,
    physics_targets, tensorize_source_actions, velocity_heading, write_prepared_clip,
)


def frame():
    def body(value):
        return {"location": dict(x=value, y=0., z=0.), "velocity": dict(x=value * 10, y=0., z=0.)}
    return {"ball": body(1.), "cars": [{"player_id": pid, **body(pid)} for pid in [3, 1, 4, 2]]}


def index(count=100):
    return {"total_samples": count, "entries": [
        {"match_id": f"synthetic-match-{i}", "shard": "synthetic.tar", "n_players": 4,
         "chunk_frames": [80, 60], "chunk_indices": [0, 3],
         "perspectives": [{"player_id": pid, "team": int(pid > 2), "frames": 140, "duration": 7.}
                          for pid in [1, 2, 3, 4]]} for i in range(count)]}


def test_match_assignment_is_stable_disjoint_and_respects_original_chunk_indices():
    original = index()
    manifest = make_split_manifest([original], revision="synthetic")
    shuffled = copy.deepcopy(original)
    shuffled["entries"].reverse()
    assert manifest == make_split_manifest([shuffled], revision="synthetic")
    assert manifest["all_splits_nonempty"]
    assert sum(manifest["match_counts"].values()) == 100
    assert len({row["match_id"] for row in manifest["matches"]}) == 100
    assert all(row["chunks"][1]["chunk_index"] == 3 for row in manifest["matches"])
    old = {row["match_id"]: row["split"] for row in manifest["matches"]}
    expanded = make_split_manifest([index(101)], revision="synthetic")
    assert all(row["split"] == old.get(row["match_id"], row["split"]) for row in expanded["matches"])


def test_duplicate_matches_cannot_leak_between_upstream_splits():
    with pytest.raises(ValueError, match="duplicate match_id"):
        make_split_manifest([index(), index()], revision="synthetic")


def test_bad_chunk_mapping_rejected():
    values = index(1)
    values["entries"][0]["chunk_indices"] = [0, 0]
    with pytest.raises(ValueError, match="uniquely"):
        make_split_manifest([values], revision="synthetic")


def test_pilot_is_excluded_and_metadata_must_be_consistent():
    data = index()
    pilots = ["synthetic-match-0", "synthetic-match-1"]
    manifest = make_split_manifest([data], revision="synthetic", pilot_match_ids=pilots)
    assert manifest["engineering_pilot_reserved"] and manifest["pilot_count"] == 2
    assert sum(manifest["match_counts"].values()) == 98
    assert all(row["split"] == "pilot" for row in manifest["matches"] if row["match_id"] in pilots)
    with pytest.raises(ValueError, match="must exist"):
        make_split_manifest([data], revision="synthetic", pilot_match_ids=["absent"])
    data["entries"][0]["perspectives"][0]["frames"] = 139
    with pytest.raises(ValueError, match="frame total"):
        make_split_manifest([data], revision="synthetic")


def test_upstream_partition_provenance_survives_merged_indices():
    first, second = index(2), index(2)
    for row in second["entries"]:
        row["match_id"] += "-second"
    sources = [{"upstream_split": "train", "index_path": "/fixture/train/index.json"},
               {"upstream_split": "test", "index_path": "/fixture/test/index.json"}]
    manifest = make_split_manifest([first, second], revision="synthetic", index_sources=sources)
    assert manifest["upstream_split_provenance_known"]
    for row in manifest["matches"]:
        expected = "test" if row["match_id"].endswith("-second") else "train"
        assert row["upstream_split"] == expected
        assert row["source_index"]["index_path"] == f"/fixture/{expected}/index.json"
    assert not make_split_manifest([first], revision="synthetic")["upstream_split_provenance_known"]


def test_physics_uses_player_identity_not_frame_list_order():
    values = physics_targets([frame()], [1, 2, 3, 4])
    assert values.shape == (1, 30)
    np.testing.assert_equal(values[0, [6, 12, 18, 24]], [1., 2., 3., 4.])
    broken = frame()
    broken["cars"][1]["player_id"] = 3
    with pytest.raises(ValueError, match="identities"):
        physics_targets([broken], [1, 2, 3, 4])


def test_normalizer_rejects_selection_and_preserves_original_units():
    x = np.asarray([[2., 7.], [4., 7.]])
    with pytest.raises(ValueError, match="discovery"):
        PhysicsNormalizer.fit(x, ["discovery", "selection"])
    normalizer = PhysicsNormalizer.fit(x, ["discovery", "discovery"])
    np.testing.assert_allclose(normalizer.transform(x), [[-1., 0.], [1., 0.]])
    np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(x)), x)
    np.testing.assert_allclose(normalizer.transform([[10., 7.]]), [[7., 0.]])


def test_invalid_and_overflowed_labels_are_rejected():
    broken = frame()
    broken["ball"]["location"]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        physics_targets([broken], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="overflowed"):
        PhysicsNormalizer.fit(np.asarray([[1e308], [-1e308]]), ["discovery", "discovery"])
    with pytest.raises(ValueError, match="two labeled"):
        PhysicsNormalizer.fit(np.zeros((2, 0)), ["discovery", "discovery"])


def test_heading_masks_stationary_and_preserves_circle():
    heading, valid = velocity_heading(np.asarray([[0., 0.], [3., 4.], [-3., 0.]]))
    np.testing.assert_equal(valid, [False, True, True])
    assert np.isnan(heading[0]).all()
    np.testing.assert_allclose(heading[1:], [[.6, .8], [-1., 0.]])


def action_audit(a, b, donor_ids=(1, 2, 3, 4)):
    return audit_future_actions(a, b, reference_player_ids=[1, 2, 3, 4],
                                donor_player_ids=donor_ids, reference_source_fps=20., donor_source_fps=20.,
                                reference_source_frame_indices=np.arange(a.shape[1]),
                                donor_source_frame_indices=np.arange(b.shape[1]) + 200)


def test_exact_future_action_audit_checks_all_players_and_no_counterfactual_claim():
    actions = np.zeros((4, 80, 9), dtype=np.uint8)
    equal = action_audit(actions, actions)
    assert equal["future_actions_exactly_equal"]
    assert not equal["controlled_counterfactual"]
    changed = actions.copy()
    changed[3, 79, 8] = 1
    report = action_audit(actions, changed)
    assert not report["future_actions_exactly_equal"]
    assert report["different_key_frame_entries"] == 1
    assert not action_audit(actions, actions, donor_ids=[4, 3, 2, 1])["future_actions_exactly_equal"]
    assert not action_audit(actions, actions[:, :-1])["future_actions_exactly_equal"]


def test_source_actions_detect_difference_hidden_by_or_downsampling():
    a = tensorize_source_actions([[{"keys": ["W"]}, {"keys": []}] for _ in range(4)])
    b = tensorize_source_actions([[{"keys": []}, {"keys": ["W"]}] for _ in range(4)])
    np.testing.assert_equal(a.max(axis=1), b.max(axis=1))
    assert not action_audit(a, b)["future_actions_exactly_equal"]
    with pytest.raises(ValueError, match="unknown action"):
        tensorize_source_actions([[{"keys": ["unknown"]}] for _ in range(4)])


def test_future_grid_compares_relative_time_not_absolute_match_time():
    actions = np.zeros((4, 3, 9), dtype=np.uint8)
    kwargs = dict(reference_player_ids=[1, 2, 3, 4], donor_player_ids=[1, 2, 3, 4],
                  reference_source_frame_indices=[10, 11, 12], donor_source_frame_indices=[210, 211, 212],
                  reference_source_fps=20., donor_source_fps=20.)
    report = audit_future_actions(actions, actions, **kwargs)
    assert report["future_actions_exactly_equal"]
    json.dumps(report)
    kwargs["donor_source_fps"] = 10.
    assert not audit_future_actions(actions, actions, **kwargs)["future_actions_exactly_equal"]
    kwargs["donor_source_frame_indices"] = [210, 212, 214]
    with pytest.raises(ValueError, match="contiguous"):
        audit_future_actions(actions, actions, **kwargs)


def test_reject_missing_player_actions():
    with pytest.raises(ValueError, match="shape"):
        action_audit(np.zeros((1, 80, 9)), np.zeros((1, 80, 9)))


def test_prepared_artifact_roundtrip_and_alignment_gate(tmp_path):
    destination = tmp_path / "synthetic.npz"
    kwargs = dict(match_id="synthetic-match-0", chunk_index=3, player_ids=[1, 2, 3, 4],
                  physics=[frame(), frame()], source_actions=np.zeros((4, 2, 9), dtype=np.uint8),
                  source_frame_indices=[240, 241], source_fps=20., source_revision="synthetic",
                  synthetic=True)
    manifest = write_prepared_clip(destination, **kwargs)
    assert manifest["synthetic"] and not manifest["research_evidence"]
    assert not manifest["video_alignment_checked"]
    assert json.loads(destination.with_suffix(".json").read_text()) == manifest
    with np.load(destination, allow_pickle=False) as data:
        np.testing.assert_equal(data["targets"], physics_targets(kwargs["physics"], kwargs["player_ids"]))
        assert data["source_actions"].shape == (4, 2, 9)
    kwargs["source_frame_indices"] = [240, 242]
    with pytest.raises(ValueError, match="contiguous"):
        write_prepared_clip(tmp_path / "bad.npz", **kwargs)
    assert not (tmp_path / "bad.npz").exists()


def prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_data_cli", Path(__file__).parents[1] / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_access_budget_and_pinned_revision_are_checked_before_download(tmp_path, monkeypatch):
    module = prepare_module()
    fake_info = SimpleNamespace(sha=module.PINNED_REVISION, gated="auto",
                                siblings=[SimpleNamespace(rfilename="test/index.json", size=101)])
    def dataset_info(*args, **kwargs):
        assert kwargs["revision"] == module.PINNED_REVISION
        return fake_info
    fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=dataset_info), get_token=lambda: None,
                               hf_hub_download=lambda *args, **kwargs: pytest.fail("budget must prevent download"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    report, path = module.check_access(tmp_path / "report.json", "test", 100, module.PINNED_REVISION, tmp_path / "raw")
    assert path is None and report["status"] == "blocked_download_budget_or_disk"
    assert not report["research_dataset_ready"]
    fake_info.sha = "incorrect"
    report, path = module.check_access(tmp_path / "report.json", "test", 1000, module.PINNED_REVISION, tmp_path / "raw")
    assert path is None and report["status"] == "blocked_access_check_error"
    assert report["error"] == {"type": "ValueError", "http_status": None}


def test_archive_member_paths_cannot_escape_data_directory():
    module = prepare_module()
    for path in ("../escape.tar", "/absolute.tar", "a/../../escape.tar", "a\\escape.tar"):
        with pytest.raises(ValueError, match="unsafe"):
            module.safe_relative(path)
    assert module.safe_relative("nested/dataset.tar") == "nested/dataset.tar"
