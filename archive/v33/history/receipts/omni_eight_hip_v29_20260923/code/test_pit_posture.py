import torch

from pit_posture import phase_weight, stance_values


def test_floor_cost_fades_near_wall_and_restores_after_climb_only_in_pit():
    gap=torch.tensor([1.,.5,.2,-1.,1.])
    platform=torch.tensor([False,False,False,True,False])
    active=torch.tensor([True,True,True,True,False])
    torch.testing.assert_close(phase_weight(gap,platform,active),torch.tensor([1.,.5,0.,1.,0.]))


def test_recorded_low_splayed_parking_is_not_accepted_as_normal_stance():
    # Recorded 0.6 m/s first outcome, 41001 update checkpoint, left front leg.
    q=torch.tensor([[-.4100,1.4484,-2.9590],[.2,.93,-1.96],[.2,.93,-1.96]])
    default=torch.tensor([[.2,.93,-1.96]]).expand_as(q)
    tolerance=torch.tensor([.25,.45,.55])
    joint,height,valid,_=stance_values(q,default,tolerance,torch.tensor([.288,.46,.30]))
    assert valid.tolist()==[False,True,False]
    assert joint[0]>0 and height[0]>0
    assert joint[1]==0 and height[1]==0


def test_stance_has_deadband_and_does_not_modify_measurements():
    q=torch.tensor([[.05,.1,.2]])
    original=q.clone()
    joint,_,valid,_=stance_values(q,torch.zeros_like(q),torch.tensor([.25,.45,.55]),torch.tensor([.42]))
    assert valid.item() and joint.item()==0
    torch.testing.assert_close(q,original)
