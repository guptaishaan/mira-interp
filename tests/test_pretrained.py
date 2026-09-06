"""Engineering contracts for the published Alakazam action-combiner adapter."""

from types import SimpleNamespace

from einops import EinopsError
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel
from mira_interp.pretrained import AlakazamFourPlayerWorldModel


@pytest.fixture
def adapter(monkeypatch):
    # Avoid constructing a codec or world model; exercise the real adapter's
    # constructor and combiner with deterministic, deliberately asymmetric values.
    def initialize_parent(self, config):
        nn.Module.__init__(self)
        self.n_players = config.n_players
        self.single_world_model = SimpleNamespace(action_encoder=SimpleNamespace(dim=3))
        self.player_embedding = nn.Parameter(torch.tensor([
            [0.0, 0.5, -0.5], [1.0, -1.0, 0.0], [-0.25, 0.0, 0.25], [0.2, 0.4, 0.6],
        ]))

    monkeypatch.setattr(MultiWrapperWorldModel, "__init__", initialize_parent)
    model = AlakazamFourPlayerWorldModel(SimpleNamespace(n_players=4))
    with torch.no_grad():
        model.player_action_projection[1].weight.copy_(torch.tensor([
            [1.0, 0.0, -0.5], [0.2, 1.0, 0.0], [0.0, -0.3, 1.5],
        ]))
        model.player_action_projection[1].bias.copy_(torch.tensor([0.1, 0.2, -0.4]))
    return model


def test_projection_then_mean_preserves_match_groups_and_player_identity(adapter):
    actions = torch.linspace(-3, 4, 8 * 2 * 3).reshape(8, 2, 3)
    result = adapter._combine_player_actions(actions)
    assert result.shape == (2, 2, 3)
    projection = adapter.player_action_projection[1]
    assert projection.weight.shape == (3, 3)  # pretrained D-to-D contract, not 4D-to-D
    expected = torch.empty_like(result)
    for match in range(2):
        for time in range(2):
            contributions = []
            for player in range(4):
                value = actions[match * 4 + player, time] + adapter.player_embedding[player]
                contributions.append(F.linear(F.silu(value), projection.weight, projection.bias))
            expected[match, time] = sum(contributions) / 4
    torch.testing.assert_close(result, expected)
    # Averaging before the nonlinear projection is a different, incorrect algorithm.
    averaged_first = actions[:4].mean(0) + adapter.player_embedding.mean(0)
    assert not torch.allclose(result[0], adapter.player_action_projection(averaged_first))
    changed = actions.clone()
    changed[3] += 2.0
    changed_result = adapter._combine_player_actions(changed)
    assert not torch.equal(changed_result[0], result[0])
    assert torch.equal(changed_result[1], result[1])
    # Permuting streams without their player embeddings must not silently erase identity.
    swapped = actions.clone()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert not torch.allclose(adapter._combine_player_actions(swapped)[0], result[0])


def test_adapter_rejects_partial_player_group_wrong_width_and_wrong_rank(adapter):
    with pytest.raises(EinopsError):
        adapter._combine_player_actions(torch.zeros(7, 2, 3))
    with pytest.raises(RuntimeError):
        adapter._combine_player_actions(torch.zeros(8, 2, 4))
    with pytest.raises(EinopsError):
        adapter._combine_player_actions(torch.zeros(2, 4, 2, 3))
