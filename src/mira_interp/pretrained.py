"""Exact Alakazam 4P action-combiner compatibility with current upstream MIRA.

Source: PyPI alakazam-mira-mini 0.1.11 wheel, SHA-256
0acea9fc285c44537785e2987a42181695500317371e6cc652cefcd38e982cfe.
The published runtime's multiplayer file differs from upstream 3d739ec only in
the projection input width and the combine method implemented below. Do not use
this adapter for original MIRA checkpoints, which concatenate player channels.
"""

from einops import rearrange
from torch import nn

from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModel


class AlakazamFourPlayerWorldModel(MultiWrapperWorldModel):
    """Apply each player's trained projection, then average over the four players."""

    def __init__(self, config):
        super().__init__(config)
        width = self.single_world_model.action_encoder.dim
        self.player_action_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def _combine_player_actions(self, a_flat):
        actions = rearrange(a_flat, "(b p) t d -> b p t d", p=self.n_players)
        actions = actions + self.player_embedding[None, :, None, :]
        return self.player_action_projection(actions).mean(dim=1)
