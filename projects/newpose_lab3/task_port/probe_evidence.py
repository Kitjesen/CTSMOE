"""History update and observable reset checks for the native interface probe."""
import torch


def append_history(history, frame, done):
    if history.ndim != 3 or frame.shape != (history.shape[0], history.shape[2]):
        raise ValueError("history/frame shapes do not match")
    if done.shape != (history.shape[0],) or done.dtype != torch.bool:
        raise ValueError("done must be a boolean environment mask")
    updated = torch.cat((history[:, 1:], frame[:, None, :]), dim=1)
    updated[done] = frame[done, None, :]
    error = torch.where(done, (updated-frame[:, None, :]).abs().amax(dim=(1,2)), 0.)
    if not torch.isfinite(updated).all() or torch.any(error != 0):
        raise RuntimeError("History contains nonfinite values or leaked pre-reset frames")
    return updated, error


def reset_verdict(steps, counts, history_errors, episode_lengths_at_reset):
    requested = steps >= 1001
    verified = (requested and bool(counts) and all(x > 0 for x in counts)
                and len(history_errors) == sum(counts)
                and len(episode_lengths_at_reset) == sum(counts)
                and all(x == 0 for x in history_errors)
                and all(x == 0 for x in episode_lengths_at_reset))
    return {"reset_verification_requested": requested, "reset_verified": bool(verified),
            "probe_passed": not requested or bool(verified),
            "reset_scope": "actual timeout/reset and history refill" if requested else "interface only; reset not required"}
