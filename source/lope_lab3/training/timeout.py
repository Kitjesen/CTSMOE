"""Timeout bootstrap for a 45-value Lope frame. Not the 53-value Thunder frame."""
import torch
from tensordict import TensorDict

from lope_lab3.spec import CRITIC_DIM, FRAME_DIM, POLICY_DIM


def terminal_timeout_reward(algorithm, rewards, dones, extras, terminated):
    if "time_outs" not in extras:
        return rewards, extras, 0
    extras = dict(extras)
    timeouts = extras.pop("time_outs").to(device=rewards.device, dtype=torch.bool).reshape(-1)
    if timeouts.shape != dones.shape or (timeouts & ~dones.bool()).any():
        raise ValueError("Timeout flags must be a subset of the current done mask")
    if not timeouts.any():
        return rewards, extras, 0
    if terminated is None:
        raise ValueError("Cannot distinguish a true termination from a timeout")
    terminated = terminated.to(device=rewards.device, dtype=torch.bool).reshape(-1)
    selected = timeouts & ~terminated
    if not selected.any():
        return rewards, extras, 0
    final = extras.get("final_obs")
    if final is None or "single_obs" not in final or "critic" not in final:
        raise ValueError("Enable compute_final_obs: terminal teacher/student frames are required")
    final_frame = final["single_obs"].to(rewards.device)
    final_critic = final["critic"].to(rewards.device)
    if final_frame.shape[-1] != FRAME_DIM or final_critic.shape[-1] != CRITIC_DIM:
        raise ValueError("Final Lope observations have incompatible shapes")
    previous = algorithm.transition.observations["policy"]
    if previous.shape[-1] != POLICY_DIM:
        raise ValueError("Missing original pre-step frame-major history")
    corrected = rewards.clone()
    teacher_count = algorithm.teacher_num_envs
    for indices, histories, teacher in (
        (algorithm.teacher_env_idxs, previous[:teacher_count], True),
        (algorithm.student_env_idxs, previous[teacher_count:], False),
    ):
        mask = selected[indices]
        chosen = indices[mask]
        if not len(chosen):
            continue
        frame = final_frame[chosen]
        terminal = TensorDict(
            {
                "policy": torch.cat((histories[mask, FRAME_DIM:], frame), dim=-1),
                "single_obs": frame,
                "critic": final_critic[chosen],
            },
            batch_size=[len(chosen)],
        )
        with torch.no_grad():
            values = algorithm.policy.evaluate(terminal, is_teacher=teacher).flatten()
        corrected[chosen] += algorithm.gamma * values
    return corrected, extras, int(selected.sum())
