import numpy as np
import torch
from mira_interp.development_capture import descriptors, channel_projection

def test_spatial_readout_preserves_view_and_bin_identity():
    x=torch.zeros(1,2,36,16,2048)
    # One localized channel in the fourth view, second latent, bottom-right bin.
    x[0,1,33:36,12:16,0]=9
    p=torch.zeros(2048,128)
    p[0,0]=1
    mean, spatial=descriptors(x,p)
    assert mean.shape==(8,2048) and spatial.shape==(8,1536)
    assert mean[7,0]==.75
    assert spatial[7,11*128]==9
    assert torch.count_nonzero(spatial)==1
    # A same-mass move changes structured descriptor but preserves spatial mean.
    y=torch.zeros_like(x)
    y[0,1,27:30,0:4,0]=9
    mean2, spatial2=descriptors(y,p)
    assert torch.equal(mean,mean2) and not torch.equal(spatial,spatial2)

def test_projection_is_fixed_orthonormal():
    a=channel_projection()
    b=channel_projection()
    assert torch.equal(a,b)
    assert torch.allclose(a.T@a,torch.eye(128),atol=1e-6,rtol=1e-6)
