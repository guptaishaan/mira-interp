"""Synthetic engineering tests; no Rocket Science observations are encoded here."""
import copy
import io

import numpy as np
import pytest

from mira_interp.clips import audit_physics_tracks, calibrate_timeline, decode_video, plan_clips, structural_validity_mask


def entry():
    return {"match_id": "synthetic", "chunk_indices": list(range(10)), "chunk_frames": [80] * 10,
            "perspectives": [{"player_id": p, "team": int(p >= 2), "frames": 800, "duration": 40., "fps": 20.,
                              "recording_offset_sec": .02 * p,
                              "anchors": [{"event_name": "KickoffEnded", "master_sec": .5}]} for p in range(4)]}


def calibration_fixture():
    e = entry()
    e["chunk_frames"] = [80]
    e["chunk_indices"] = [0]
    goals = [{"event_name": "GoalScored", "master_sec": 8.133 + frame / 20} for frame in [20, 40, 60]]
    for p in e["perspectives"]:
        p.update(frames=80, duration=4., anchors=copy.deepcopy(goals))
    tracks = [[{"score_blue": t // 20, "score_orange": 0, "time_remaining": 300. - t / 20}
               for t in range(80)] for _ in range(4)]
    return e, tracks


def test_score_anchors_recover_missing_trim_origin_and_hash_inputs():
    e, tracks = calibration_fixture()
    result = calibrate_timeline(e, tracks)
    assert not result["uses_position_or_velocity_values"]
    for view in result["views"]:
        assert view["calibrated_origin_seconds"] == pytest.approx(8.133)
        assert len(view["goal_pairs"]) == 3
        assert view["max_absolute_residual_seconds"] < 1e-12
        assert len(view["score_track_sha256"]) == 64


def test_timeline_ambiguity_and_insufficient_events_fail_without_guessing():
    e, tracks = calibration_fixture()
    tracks[0][20]["score_blue"] = 3
    with pytest.raises(ValueError, match="unambiguously"):
        calibrate_timeline(e, tracks)
    e, tracks = calibration_fixture()
    for state in tracks[0][60:]:
        state["score_blue"] = 2
    with pytest.raises(ValueError, match="fewer than three"):
        calibrate_timeline(e, tracks)
    e, tracks = calibration_fixture()
    e["perspectives"][0]["anchors"][0]["master_sec"] += .3
    with pytest.raises(ValueError, match="0.1 second"):
        calibrate_timeline(e, tracks)


def test_nonzero_initial_score_maps_to_goal_number_without_renumbering():
    e, tracks = calibration_fixture()
    for perspective in e["perspectives"]:
        perspective["anchors"].insert(0, {"event_name": "GoalScored", "master_sec": 5.})
    for track in tracks:
        for state in track:
            state["score_blue"] += 1
    result = calibrate_timeline(e, tracks)
    assert result["views"][0]["goal_pairs"][0]["score_total"] == 2
    assert result["views"][0]["calibrated_origin_seconds"] == pytest.approx(8.133)


def test_clip_plan_uses_calibrated_events_and_rejects_gapped_source_times():
    e = entry()
    first = plan_clips(e, calibrated_origins=[0.] * 4)
    assert len(first) == 8 and len({p["source_start_frame"] for p in first}) == 8
    assert all(p["source_start_frame"] > 20 for p in first)
    assert first == plan_clips(e, calibrated_origins=[0.] * 4)
    with pytest.raises(ValueError, match="verified calibrated"):
        plan_clips(e)
    e["chunk_indices"][3] = 12
    with pytest.raises(ValueError, match="gapped"):
        plan_clips(e, calibrated_origins=[0.] * 4)


def test_per_view_physics_preserves_labels_and_rejects_frozen_replay():
    physics = []
    def body(x):
        return {"location": {"x": x, "y": 0, "z": 100}, "velocity": {"x": 200, "y": 0, "z": 0}}
    for view in range(4):
        physics.append([{"ball": body(t * 10 + view), "cars": [
            {"player_id": p, "team": int(p >= 2), "is_local": p == view, "attacker_player_id": -1,
             **body(p * 100 + t * 10)} for p in range(4)]} for t in range(16)])
    actions = [[{"keys": []}] * 16 for _ in range(4)]
    y, a, report = audit_physics_tracks(physics, actions, list(range(4)), [0, 0, 1, 1])
    assert y.shape == (4, 16, 30) and a.shape == (4, 16, 9)
    np.testing.assert_equal(y[:, 0, 0], [0, 1, 2, 3])
    assert not report["exact_cross_view_synchronization"]
    physics[0][3]["cars"][0]["attacker_player_id"] = 2
    mask, flags = structural_validity_mask(physics[0], list(range(4)), [0, 0, 1, 1], 0)
    assert not mask[3] and flags["demolition_frames"] == 1
    physics[0][3]["cars"][0]["attacker_player_id"] = -1
    for view in range(4):
        physics[view][1] = copy.deepcopy(physics[view][0])
    with pytest.raises(ValueError, match="frozen"):
        audit_physics_tracks(physics, actions, list(range(4)), [0, 0, 1, 1])


def test_pyav_decode_checks_frame_count_pts_and_upstream_resize():
    av = pytest.importorskip("av")
    stream_buffer = io.BytesIO()
    with av.open(stream_buffer, mode="w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((16, 32, 3), i * 50, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    blob = stream_buffer.getvalue()
    pixels, times, audit = decode_video(blob, [0, 2], expected_frames=3, size=(8, 16))
    assert pixels.shape == (2, 3, 8, 16) and pixels.dtype == np.uint8
    np.testing.assert_allclose(times - times[0], [0., .1])
    assert audit["pts_monotonic_and_uniform"]
    with pytest.raises(ValueError, match="decoded video count"):
        decode_video(blob, [0, 2], expected_frames=4, size=(8, 16))
