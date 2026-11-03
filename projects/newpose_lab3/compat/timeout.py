"""Restore old NEWPOSE timeout semantics at the upstream process-step boundary."""

import torch
from tensordict import TensorDict


def terminal_timeout_reward(algorithm, rewards, dones, extras, terminated):
    """Add gamma*V(s_T) for time-limit-only resets; remove upstream V(s_t) path.

    Lab3 final_obs is computed before reset with update_history=False. Its
    history therefore does not include the final frame. The upstream transition
    already owns the pre-step history in teacher-then-student order; advance
    that history once using final_obs.single_obs, just as old History.next did.
    """
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
    if terminated.shape != timeouts.shape:
        raise ValueError("Termination mask differs from timeout batch")
    selected = timeouts & ~terminated
    if not selected.any():
        return rewards, extras, 0
    final = extras.get("final_obs")
    if final is None or "single_obs" not in final or "critic" not in final:
        raise ValueError("Enable compute_final_obs: terminal teacher/student frames are required")
    final_frame = final["single_obs"].to(rewards.device)
    final_critic = final["critic"].to(rewards.device)
    if final_frame.shape != (len(rewards), 53) or final_critic.shape[0] != len(rewards):
        raise ValueError("Final observations have incompatible shapes")
    if not torch.isfinite(final_frame[selected]).all() or not torch.isfinite(final_critic[selected]).all():
        raise ValueError("Nonfinite terminal observations")
    previous = algorithm.transition.observations["policy"]
    if previous.shape != (len(rewards), 530):
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
        terminal = TensorDict({"policy": torch.cat((histories[mask, 53:], frame), dim=-1),
                               "single_obs": frame, "critic": final_critic[chosen]},
                              batch_size=[len(chosen)])
        with torch.no_grad():
            values = algorithm.policy.evaluate(terminal, is_teacher=teacher).flatten()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite terminal bootstrap value")
        corrected[chosen] += algorithm.gamma * values
    return corrected, extras, int(selected.sum())
