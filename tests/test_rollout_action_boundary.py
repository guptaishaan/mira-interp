"""Exercise the real inference adapter at the upstream integer-embedding boundary."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import rollout_steering as run
from mira.world_model.actions_config import ActionConfig


def test_source_byte_actions_reach_upstream_embedding_without_changing_values(monkeypatch):
    source=torch.arange(4*24*9,dtype=torch.int32).remainder(2).to(torch.uint8).reshape(4,24,9)
    original=source.clone()
    embedding=torch.nn.Embedding(2,3)
    config=ActionConfig(valid_keys=[f"key{i}" for i in range(9)])
    class ReachedUpstream(Exception):pass
    class Model:
        def __init__(self):self.config=SimpleNamespace(actions=config)
        def inference(self,batch,*args,**kwargs):
            assert batch.actions.key_presses.dtype==torch.int32
            assert torch.equal(embedding(batch.actions.key_presses),embedding(source.long()))
            assert torch.equal(batch.actions.key_presses.float(),source.float())
            raise ReachedUpstream()
    monkeypatch.setattr(run.VideoActionBatch,"to",lambda self,*args,**kwargs:self)
    with pytest.raises(ReachedUpstream):
        run.infer(Model(),torch.zeros(4,24,3,1,1),source,2026090701,[],None,hooked=False)
    assert torch.equal(source,original) and source.dtype==torch.uint8
