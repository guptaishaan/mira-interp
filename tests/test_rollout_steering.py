"""CPU path math and mocked real-shaped sampler calls; no rollout results."""
import numpy as np
import pytest
import torch

from mira.world_model.schedule import build_inference_schedule
from mira_interp.probes import RidgeModel
from mira_interp.rollout_steering import (FirstGeneratedEdit, HEIGHT_SUPPORT, clipped_heights,
    conditions, forward_map, hidden_video, path_delta)
from mira_interp.sparse_causal_fidelity import spatial_lift


def fixture():
    rng = np.random.default_rng(101)
    probe = RidgeModel(.1, rng.normal(size=1536), rng.uniform(.5, 2, 1536), rng.normal(size=15),
                       rng.uniform(.5, 2, 15), rng.normal(size=(1536, 15)))
    maps = {"physical_mean": np.array(500.), "physical_scale": np.array(400.)}
    for name, dim in (("affine", 1), ("quadratic", 2)):
        maps.update({f"{name}__x_mean": np.zeros(dim), f"{name}__x_scale": np.ones(dim),
                     f"{name}__y_mean": rng.normal(size=1536), f"{name}__y_scale": np.ones(1536),
                     f"{name}__coefficient": rng.normal(size=(dim, 1536))})
    q = torch.zeros(2048, 128)
    q[:128] = torch.eye(128)
    return probe, maps, q


def test_hidden_future_clone_and_condition_count():
    source = torch.full((4, 24, 3, 2, 2), 99., dtype=torch.float32)
    zero, alternate = hidden_video(source), hidden_video(source, placeholder=255.)
    assert torch.equal(zero[:, :16], source[:, :16])
    assert torch.equal(alternate[:, :16], source[:, :16])
    assert torch.count_nonzero(zero[:, 16:]) == 0
    assert torch.all(alternate[:, 16:] == 255)
    assert torch.all(source == 99)
    assert len(conditions()) == len(set(conditions())) == 21
    assert sum(dose == 0 for _, dose in conditions()) == 1


def test_clipped_coordinate_paths_probe_dose_and_control_norms():
    probe, maps, q = fixture()
    edge = clipped_heights(-100., -600.)
    assert edge["base_height_clipped"] == HEIGHT_SUPPORT[0] and edge["effective_dose"] == 0
    for dose in (-300., 0., 300.):
        deltas = {name: path_delta(name, 600., dose, probe, maps, q, seed=2)[0]
                  for name in ("probe_linear", "affine_forward", "quadratic_forward", "random_norm", "wrong_variable_ball_x")}
        measured_shift = probe.predict(deltas["probe_linear"].numpy())[0, 2] - probe.predict(np.zeros((1, 1536)))[0, 2]
        assert measured_shift == pytest.approx(dose, abs=1e-4)
        expected = forward_map(maps, "quadratic", 600.+dose)-forward_map(maps, "quadratic", 600.)
        np.testing.assert_allclose(deltas["quadratic_forward"], expected, atol=1e-6)
        reference = torch.linalg.vector_norm(spatial_lift(deltas["quadratic_forward"], q))
        for name in ("random_norm", "wrong_variable_ball_x"):
            assert float(torch.linalg.vector_norm(spatial_lift(deltas[name], q))) == pytest.approx(float(reference), rel=2e-6)
        if dose == 0:
            assert all(torch.count_nonzero(delta) == 0 for delta in deltas.values())


def test_forward_map_difference_uses_both_physical_and_ridge_scales():
    probe, maps, q = fixture()
    maps["affine__coefficient"][:] = 3
    maps["affine__x_mean"][:] = 7
    maps["affine__x_scale"][:] = 4
    maps["affine__y_scale"][:] = 2
    delta, _ = path_delta("affine_forward", 600., 300., probe, maps, q, seed=2)
    torch.testing.assert_close(delta, torch.full((1, 1536), 300/400/4*3*2))


class Block(torch.nn.Module):
    def forward(self, value):
        return value, "cache auxiliary unchanged"


class TinySamplerBoundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.ModuleList([Block() for _ in range(16)])

    def forward(self, z_t, a, tau, tau_delta=None, return_kv=False, kv_caches=None, clean_past=None):
        value = z_t.expand(*z_t.shape[:-1], 2048)
        # Only hooked final block needs execution for this boundary fixture.
        result = self.transformer[15](value)
        assert result[1] == "cache auxiliary unchanged"
        return result[0]


def replay(module, schedule, dtype=torch.float32):
    actions = torch.ones(1, 1, 2)
    outputs = []
    module(torch.zeros(1, 8, 36, 16, 1, dtype=dtype), actions, torch.ones(1, 8, 1, 1, 1, dtype=dtype), return_kv=True)
    for latent in range(4):
        for tau in schedule[:-1]:
            outputs.append(module(torch.zeros(1, 1, 36, 16, 1, dtype=dtype), actions, torch.full((1, 1, 1, 1, 1), tau, dtype=dtype), kv_caches=[("k", "v")]))
        module(torch.zeros(1, 1, 36, 16, 1, dtype=dtype), actions, torch.ones(1, 1, 1, 1, 1, dtype=dtype), return_kv=True, kv_caches=[("k", "v")])
    return outputs


def test_hook_real_tau_single_first_latent_tuple_locality_and_cleanup():
    torch.set_num_threads(4)
    schedule = build_inference_schedule(10, torch.device("cpu"), "linear_quadratic").tolist()
    assert .5 not in schedule and schedule[8] == pytest.approx(.527789294719696)
    _, _, q = fixture()
    module = TinySamplerBoundary()
    with FirstGeneratedEdit(module, schedule=schedule, descriptor_delta=torch.ones(1, 1536), projection=q) as edit:
        outputs = replay(module, schedule)
        edit.validate()
    assert edit.hits == 1 and len(edit.trace) == 45
    assert torch.count_nonzero(outputs[8][0, 0, :9]) > 0
    assert torch.count_nonzero(outputs[8][0, 0, 9:]) == 0
    assert all(torch.count_nonzero(output) == 0 for i, output in enumerate(outputs) if i != 8)
    assert not module._forward_pre_hooks and not module.transformer[15]._forward_hooks
    with FirstGeneratedEdit(module, schedule=schedule, descriptor_delta=torch.zeros(1, 1536), projection=q) as noop:
        assert all(torch.count_nonzero(output) == 0 for output in replay(module, schedule))
        noop.validate()
    with FirstGeneratedEdit(module, schedule=schedule) as bf16:
        replay(module, schedule, torch.bfloat16)
        bf16.validate()
    target = next(call for call in bf16.trace if call["target_call"])
    assert target["tau"] == .52734375 and target["nominal_schedule_tau"] == schedule[8]
    with pytest.raises(ValueError, match="fresh8-latent"):
        with FirstGeneratedEdit(module, schedule=schedule):
            module(torch.zeros(1, 1, 36, 16, 1), torch.ones(1, 1, 2), torch.zeros(1, 1, 1, 1, 1), kv_caches=[])
    assert not module._forward_pre_hooks and not module.transformer[15]._forward_hooks
