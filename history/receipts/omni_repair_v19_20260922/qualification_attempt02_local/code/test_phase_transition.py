import numpy as np
import pytest
import torch

from phase_landing import capped_vertices
from phase_transition import Recovery, approach_gate


def test_approach_release_precedes_old_quarter_meter_gate():
    torch.testing.assert_close(approach_gate(torch.tensor([1.1,1.,.75,.5,.1])),
                               torch.tensor([1.,1.,.5,0.,0.]))


def test_recovery_requires_support_and_rises_over_eight_tenths():
    state=Recovery(2,'cpu')
    state.update(torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),.02,torch.ones(2,dtype=torch.bool))
    for _ in range(50):
        state.update(torch.ones(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),.02,torch.ones(2,dtype=torch.bool))
    assert state.weight.sum()==0
    previous=state.weight.clone()
    for _ in range(46):
        state.update(torch.ones(2),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),.02,torch.ones(2,dtype=torch.bool))
        assert bool((state.weight-previous <= .025001).all())
        previous=state.weight.clone()
    torch.testing.assert_close(state.weight,torch.tensor([1.,0.]))


def test_loss_of_contact_does_not_release_already_restored_pose():
    state=Recovery(1,'cpu')
    state.update(torch.ones(1),torch.tensor([False]),torch.tensor([False]),.02,torch.tensor([False]))
    assert state.weight.item()==1.
    state.update(torch.zeros(1),torch.tensor([False]),torch.tensor([False]),.02,torch.tensor([False]))
    assert state.weight.item()==0.


@pytest.mark.parametrize('sign,kind',[(1,'stairs_up'),(-1,'stairs_down')])
def test_mesh_landing_preserves_inner_steps_and_center(sign,kind):
    x=np.array([-4.5,-3.5,-2.8,-2.,0.,2.,2.8,3.5,4.5])
    vertices=np.stack((x,np.zeros_like(x),sign*np.maximum(np.floor((np.abs(x)-1.5)/.31),0)*.2),axis=1)
    result=capped_vertices(vertices,kind)
    np.testing.assert_allclose(result[np.abs(x)<2.7],vertices[np.abs(x)<2.7])
    assert len(np.unique(result[np.abs(x)>=2.8,2]))==1
    np.testing.assert_array_equal(vertices[:,:2],result[:,:2])


def test_invalid_mesh_cannot_silently_change_stair_geometry():
    with pytest.raises(ValueError):
        capped_vertices(np.zeros((3,3)),'stairs_up')
