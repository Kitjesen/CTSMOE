import torch
import pytest

from assess_v33 import teacher_mean_actions


class FakePolicy:
    state_dependent_std = True

    def critic_obs_normalizer(self, value):
        return value + 1

    def get_critic_obs(self, obs):
        return obs["critic"]

    def single_obs_normalizer(self, value):
        return value + 2

    def teacher_encoder(self, value):
        return value[:, :1]

    def actor(self, value):
        # [mean, std] axis must be reduced by the helper.
        return (torch.stack((value[:, :2], value[:, :2] + 100), dim=1)
                if self.state_dependent_std else value[:, :2])


@pytest.mark.parametrize('state_dependent', [True, False])
def test_teacher_mean_actions_selects_ids_and_mean_head(state_dependent):
    policy = FakePolicy()
    policy.state_dependent_std = state_dependent
    obs = {"critic": torch.arange(12.).reshape(3, 4),
           "single_obs": torch.arange(15.).reshape(3, 5)}
    result = teacher_mean_actions(policy, obs, torch.tensor([2, 0]))
    assert result.shape == (2, 2)
    expected = torch.cat((obs["critic"][[2, 0], :1] + 1,
                            obs["single_obs"][[2, 0], :1] + 2), dim=-1)
    torch.testing.assert_close(result, expected)
