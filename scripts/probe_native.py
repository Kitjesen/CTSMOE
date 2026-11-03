"""Probe the integrated official Thunder environment without an optimizer.

python scripts/probe_native.py --urdf /abs/thunder_v4.urdf --output /abs/fresh-probe \
    --num-envs 32 --steps 16 --visualizer none --device cuda:0
Use --fresh for the from-scratch task, --steps 1002 for reset/history verification,
or --policy for a compatible 530-to-16 JIT.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go2-root", type=Path, default=project / "upstream" / "go2_rl_lab")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--fresh", action="store_true", help="Probe the independent from-scratch task")
    preliminary, _ = parser.parse_known_args()
    go2 = preliminary.go2_root.resolve()
    sys.path[:0] = [str(project / "source"), str(go2 / "source" / "rsl_rl"), str(go2 / "source" / "go2_rl_lab")]
    # The kitless importer's bundled USD schemas must register before Lab's first USD lookup.
    from isaacsim.asset.importer.urdf import URDFImporter  # noqa: F401
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 4 or args.num_envs % 4:
        parser.error("Require positive steps and num-envs divisible by four")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["NEWPOSE_URDF"] = str(args.urdf.resolve())
    os.environ["NEWPOSE_USD_CACHE"] = str(output / "converted")
    os.environ.pop("NEWPOSE_CANDIDATE_OVERRIDES", None)

    import gymnasium as gym
    import numpy as np
    import torch
    from thunder_lab3.tasks.thunder import TASK_ID, FRESH_TASK_ID
    from thunder_lab3.tasks.thunder.env_cfg import ThunderNewposeEnvCfg, ThunderFreshEnvCfg
    from thunder_lab3.training.history import validate_observations
    from thunder_lab3.tasks.thunder.mdp.observations import verify_student_contract
    from probe_evidence import append_history, reset_verdict
    from go2_rl_lab.utils.terrains import get_assigned_terrain_masks
    from thunder_lab3.tasks.thunder.mdp.privileged import JOINTS, WHEEL_BODIES

    task_id = FRESH_TASK_ID if args.fresh else TASK_ID
    cfg = ThunderFreshEnvCfg() if args.fresh else ThunderNewposeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    policy = torch.jit.load(str(args.policy), map_location=args.device).eval() if args.policy else None
    records = []
    terminal_records = []
    history_errors = []
    reset_history_errors = []
    reset_episode_lengths = []
    reset_counts = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    natural_timeouts = torch.zeros_like(reset_counts)
    forced_timeouts = torch.zeros_like(reset_counts)
    timeout_injections = []
    result = {"completed": False, "training_started": False, "optimizer_updates": 0,
              "steps_requested": args.steps, "seed": cfg.seed, "num_envs": args.num_envs,
              "policy": str(args.policy.resolve()) if args.policy else None,
              "action_source": "external JIT" if args.policy else "zero actions", "task": task_id}

    def terrain_snapshot(env):
        terrain = env.scene.terrain
        levels = terrain.terrain_levels
        if bool(((levels < 0) | (levels > 9)).any()):
            raise RuntimeError("Terrain levels escaped L0..L9")
        return {name: {"environments": int(mask.sum()),
                       "level_counts_L0_to_L9": torch.bincount(levels[mask], minlength=10).cpu().tolist()}
                for name, mask in get_assigned_terrain_masks(env).items()}

    with launch_simulation(cfg.sim, args):
        env = gym.make(task_id, cfg=cfg)
        base = env.unwrapped
        try:
            obs, _ = env.reset()
            result["student_contract"] = verify_student_contract(base, fresh=args.fresh)
            obs, _ = env.reset()
            validate_observations(obs)
            expected_history = obs["single_obs"][:, None].repeat(1, 10, 1)
            torch.testing.assert_close(obs["policy"].reshape(args.num_envs, 10, 53), expected_history)
            result["initial_terrain"] = terrain_snapshot(base)
            result["reward_terms"] = [{"name": name, "weight": base.reward_manager.get_term_cfg(name).weight}
                                      for name in base.reward_manager.active_terms]
            result["termination_terms"] = list(base.termination_manager.active_terms)
            robot, contact = base.scene["robot"], base.scene["contact_forces"]
            joint_ids = [robot.joint_names.index(name) for name in JOINTS]
            contact_names = list(contact.body_names)
            wheel_ids = [contact_names.index(name) for name in WHEEL_BODIES]
            leg_ids = [i for i, name in enumerate(contact_names) if "_thigh_" in name or "_calf_" in name]
            result["joint_names"] = list(JOINTS)
            result["contact_body_names"] = contact_names
            leg_sensor_cfg = base.termination_manager.get_term_cfg("leg_contact").params["sensor_cfg"]
            result["leg_termination_sensor_selection"] = [contact_names[i] for i in leg_sensor_cfg.body_ids]
            result["episode_length_s"] = cfg.episode_length_s
            result["max_episode_steps"] = base.max_episode_length
            result["physics_configuration"] = {
                "use_mujoco_contacts": bool(base.cfg.sim.physics.solver_cfg.use_mujoco_contacts),
                "physics_dt_s": base.cfg.sim.dt, "control_decimation": base.cfg.decimation,
                "contact_period_s": base.cfg.scene.contact_forces.update_period,
                "contact_history_samples": base.cfg.scene.contact_forces.history_length,
            }
            result["backend"] = type(robot).__module__
            if not result["backend"].startswith("isaaclab_newton"):
                raise RuntimeError("The integrated task did not instantiate the Newton backend")
            original_reset_idx = base._reset_idx

            def capture_terminal_reset(env_ids):
                # Capture BEFORE autoreset clears sensor history and replaces the terminal pose.
                ids = torch.as_tensor(env_ids, device=base.device, dtype=torch.long)
                state = base._leg_contact
                terminal = {
                    "step": torch.full_like(ids, step), "env_ids": ids,
                    "episode_length_before_reset": base.episode_length_buf[ids],
                    "termination_terms": torch.stack([base.termination_manager.get_term(name)[ids]
                        for name in result["termination_terms"]], dim=-1),
                    "root_pos_w": robot.data.root_pos_w.torch[ids],
                    "root_relative_env": (robot.data.root_pos_w.torch-base.scene.env_origins)[ids],
                    "root_quat_xyzw": robot.data.root_quat_w.torch[ids],
                    "gravity_b": robot.data.projected_gravity_b.torch[ids],
                    "joint_pos": robot.data.joint_pos.torch[ids][:, joint_ids],
                    "joint_vel": robot.data.joint_vel.torch[ids][:, joint_ids],
                    "contact_normal": contact.data.net_normal_forces_w.torch[ids],
                    "contact_normal_history_peak_per_body": contact.data.net_normal_forces_w_history.torch[ids].norm(dim=-1).amax(dim=1),
                    "leg_normal_history_newest_first": contact.data.net_normal_forces_w_history.torch[ids][:, :, leg_ids],
                    "terrain_levels": base.scene.terrain.terrain_levels[ids],
                    "terrain_types": base.scene.terrain.terrain_types[ids],
                    "leg_contact_duration_before": state["duration"][ids],
                    "leg_contact_failed_before": state["failed"][ids],
                    "leg_contact_touched_before": state["touched"][ids],
                }
                row = {name: value.detach().cpu().numpy().copy() for name, value in terminal.items()}
                original_reset_idx(env_ids)
                for name in ("duration", "failed", "touched"):
                    row["leg_contact_" + name + "_after"] = base._leg_contact[name][ids].detach().cpu().numpy().copy()
                terminal_records.append(row)

            base._reset_idx = capture_terminal_reset
            for step in range(args.steps):
                injected = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                if args.steps >= 1002 and step == args.steps - 2:
                    # Test-only counter injection: exercise the real timeout/reset path in every remaining env.
                    injected = natural_timeouts == 0
                    base.episode_length_buf[injected] = base.max_episode_length - 1
                    timeout_injections.append({"step": step, "env_ids": injected.nonzero().flatten().cpu().tolist()})
                with torch.inference_mode():
                    action = policy(obs["policy"]) if policy else torch.zeros((args.num_envs, 16), device=args.device)
                if action.shape != (args.num_envs, 16) or not torch.isfinite(action).all():
                    raise RuntimeError("Invalid 16-action command")
                obs, reward, terminated, truncated, _ = env.step(action)
                validate_observations(obs)
                done = terminated | truncated
                timeout = base.termination_manager.get_term("time_out").clone()
                natural_timeouts += (timeout & ~injected).long()
                forced_timeouts += (timeout & injected).long()
                reset_counts += done.long()
                expected_history, _ = append_history(expected_history, obs["single_obs"], done)
                actual_history = obs["policy"].reshape(args.num_envs, 10, 53)
                chronological_error = (actual_history - expected_history).abs().amax(dim=(1, 2))
                reset_error = (actual_history - obs["single_obs"][:, None]).abs().amax(dim=(1, 2))
                history_errors.append(float(chronological_error.max()))
                reset_history_errors.extend(reset_error[done].cpu().tolist())
                reset_episode_lengths.extend(base.episode_length_buf[done].cpu().tolist())
                if bool((chronological_error > 1.e-6).any()):
                    raise RuntimeError(f"Environment history differs from the chronological/reset contract at step {step}")
                values = {
                    "policy530": obs["policy"], "single_obs53": obs["single_obs"], "critic309": obs["critic"],
                    "actions": action, "reward": reward, "done": done, "terminated": terminated, "truncated": truncated,
                    "time_out_term": timeout, "forced_timeout_injection": injected,
                    "termination_terms": torch.stack([base.termination_manager.get_term(name)
                        for name in result["termination_terms"]], dim=-1),
                    "episode_length_after_step": base.episode_length_buf,
                    "history_max_abs_error": chronological_error, "reset_refill_max_abs_error": reset_error,
                    "terrain_levels": base.scene.terrain.terrain_levels, "terrain_types": base.scene.terrain.terrain_types,
                    "root_pos_w": robot.data.root_pos_w.torch, "root_quat_xyzw": robot.data.root_quat_w.torch,
                    "root_lin_vel_b": robot.data.root_lin_vel_b.torch, "root_ang_vel_b": robot.data.root_ang_vel_b.torch,
                    "joint_pos": robot.data.joint_pos.torch[:, joint_ids], "joint_vel": robot.data.joint_vel.torch[:, joint_ids],
                    "joint_acc": robot.data.joint_acc.torch[:, joint_ids], "torque_proxy": robot.actuators.applied_effort.torch[:, joint_ids],
                    "contact_normal": contact.data.net_normal_forces_w.torch, "contact_total": contact.data.net_forces_w.torch,
                    "commands": base.command_manager.get_command("base_velocity"),
                    "reward_term_weighted_rates": base.reward_manager._step_reward,
                }
                if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                    raise RuntimeError(f"Nonfinite native evidence at step {step}")
                records.append({name: value.detach().cpu().numpy().copy() for name, value in values.items()})
            result["final_terrain"] = terrain_snapshot(base)
            result["completed"] = True
        except Exception as exc:
            result["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            result["steps_recorded"] = len(records)
            result["reset_count_per_env"] = reset_counts.cpu().tolist()
            result["natural_timeouts_per_env"] = natural_timeouts.cpu().tolist()
            result["injected_timeouts_per_env"] = forced_timeouts.cpu().tolist()
            result["timeout_test_injections"] = timeout_injections
            result["history_max_abs_error"] = max(history_errors, default=None)
            result.update(reset_verdict(args.steps, result["reset_count_per_env"], reset_history_errors, reset_episode_lengths))
            result["all_env_timeout_path_verified"] = bool(((natural_timeouts + forced_timeouts) > 0).all())
            result["probe_passed"] = bool(result["completed"] and result["probe_passed"]
                and (args.steps < 1002 or result["all_env_timeout_path_verified"]))
            result["interface_passed"] = result["probe_passed"]
            if records:
                data = {name: np.stack([row[name] for row in records]) for name in records[0]}
                np.savez_compressed(output / "trajectory.npz", **data)
                force = np.linalg.norm(data["contact_normal"], axis=-1)
                result["joint_statistics"] = {name: {"min": value.min(axis=(0, 1)).tolist(),
                    "max": value.max(axis=(0, 1)).tolist(), "rms": np.sqrt(np.mean(value**2, axis=(0, 1))).tolist()}
                    for name, value in data.items() if name in ("joint_pos", "joint_vel", "joint_acc", "torque_proxy")}
                result["sampled_contact_normal_peaks_N"] = {"wheels": float(force[..., wheel_ids].max()),
                    "thigh_calf": float(force[..., leg_ids].max()) if leg_ids else None}
                result["teacher_feature_ranges"] = {name: {"min": float(data["critic309"][..., slots].min()),
                    "max": float(data["critic309"][..., slots].max())}
                    for name, slots in (("joint_acc16", slice(273,289)), ("torque_proxy16",slice(289,305)), ("wheel_normal4",slice(305,309)))}
                ordinary = ~data["forced_timeout_injection"]
                every_step = ((data["done"] | ~ordinary).all(axis=0) & ordinary.any(axis=0))
                streak = np.zeros(args.num_envs, dtype=np.int64)
                maximum_streak = streak.copy()
                for done_row, injected_row in zip(data["done"], data["forced_timeout_injection"]):
                    streak = np.where(done_row & ~injected_row, streak + 1, 0)
                    maximum_streak = np.maximum(maximum_streak, streak)
                result["rollout_diagnostics"] = {
                    "reset_every_ordinary_step_env_ids": np.flatnonzero(every_step).tolist(),
                    "max_consecutive_reset_steps_per_env": maximum_streak.tolist(),
                    "invalid_repeated_one_step_reset_env_ids": np.flatnonzero(maximum_streak >= 3).tolist(),
                    "termination_term_counts": dict(zip(result["termination_terms"], data["termination_terms"].sum(axis=(0,1)).tolist())),
                    "post_reset_sensor_values_are_censored": bool(data["done"].any()),
                }
                result["rollout_valid"] = bool(not (maximum_streak >= 3).any())
                result["probe_passed"] = bool(result["interface_passed"] and result["rollout_valid"])
            if terminal_records:
                terminal_data = {name: np.concatenate([row[name] for row in terminal_records], axis=0)
                                 for name in terminal_records[0]}
                np.savez_compressed(output / "terminal_pre_reset.npz", **terminal_data)
                peaks = terminal_data["contact_normal_history_peak_per_body"]
                result["terminal_pre_reset_summary"] = {
                    "episodes": len(terminal_data["env_ids"]),
                    "normal_history_peak_N_by_body": dict(zip(contact_names, peaks.max(axis=0).tolist())),
                    "root_relative_z_min": float(terminal_data["root_relative_env"][:,2].min()),
                    "leg_failed_still_true_after_reset_count": int(terminal_data["leg_contact_failed_after"].sum()),
                    "leg_duration_max_after_reset_s": float(terminal_data["leg_contact_duration_after"].max()),
                    "termination_term_counts": dict(zip(result["termination_terms"], terminal_data["termination_terms"].sum(axis=0).tolist())),
                }
            result["scope"] = "50 Hz interface/reward/history/reset evidence; injected timeouts are not survival, and sampled peaks do not resolve physics-step impacts. No optimizer or formal continuation."
            (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            env.close()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["probe_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
