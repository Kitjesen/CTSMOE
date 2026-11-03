"""First-episode staircase traces from a frozen student; no optimizer updates."""

import argparse
import json
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(STAGE / "orchestration"), str(STAGE / "code")]
from launch_common import args as train_args, frozen, digest  # noqa: E402
from state_guard import state_sha256  # noqa: E402


def recording(checkpoint, output):
    def learn(self, updates, init_at_random_ep_len=False):
        import numpy as np
        import torch
        import types

        assert updates == 0 and not init_at_random_ep_len
        raw = self.env.unwrapped
        raw.cfg.episode_length_s = 32.0  # Zero-update recording only; training keeps 25s.
        robot = raw.scene["robot"]
        terrain = raw.scene.terrain
        command = raw.command_manager.get_term("base_velocity")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        flat_cols = [i for i, name in enumerate(saved["terrain_column_names"]) if name == "flat"]
        assert flat_cols
        cases = [dict(column=flat_cols[len(flat_cols)//2], name="flat_long", level=2, side=False)]
        ids = self.alg.student_env_idxs[: len(cases)]
        before = {
            k: state_sha256(v, torch)
            for k, v in (
                ("model", self.alg.policy.state_dict()),
                ("ppo", self.alg.optimizer.state_dict()),
                ("student", self.alg.optimizer_stu_enc.state_dict()),
            )
        }
        raw.curriculum_manager.compute = lambda env_ids: None
        for i, c in zip(ids, cases):
            terrain.terrain_types[i] = c["column"]
            terrain.terrain_levels[i] = c["level"]
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        raw.scene.env_origins[:] = terrain.env_origins
        command._init_terrain_infos()
        command._update_env_command_ranges()
        command.refresh_membership()
        torch.manual_seed(240924)
        raw.reset()
        state = robot.data.default_root_state[ids].clone()
        origins = terrain.env_origins[ids].clone()
        state[:, :3] += origins
        state[:, 7:] = 0
        side = torch.tensor([c["side"] for c in cases], device=raw.device)
        yaw = torch.where(side, -torch.pi / 2, 0.0)
        state[:, 3:7] = 0
        state[:, 3] = torch.cos(yaw / 2)
        state[:, 6] = torch.sin(yaw / 2)
        robot.write_root_state_to_sim(state, env_ids=ids)
        q = robot.data.default_joint_pos[ids].clone()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        robot.set_joint_position_target(q, env_ids=ids)
        robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=ids)
        raw.stair_axis[ids] = 0
        raw.stair_motion_sign[ids] = 1
        raw.episode_length_buf[ids] = 0
        original_update = command._update_command
        tick = [0]

        def scripted(_self):
            original_update()
            command.commands[ids] = 0
            if tick[0] >= 100:
                command.commands[ids[~side], 0] = 1.0
                command.commands[ids[side], 1] = 0.25

        command._update_command = types.MethodType(scripted, command)
        scripted(command)
        original_reset = raw._reset_idx

        def reset_other(env_ids):
            other = env_ids[~torch.isin(env_ids, ids)]
            if len(other):
                original_reset(other)

        raw._reset_idx = reset_other
        output.mkdir(exist_ok=False)
        mesh = terrain._showcase_imported_mesh
        # Extract each actual atlas tile; one independent visual mesh per clip.
        for n, (c, o) in enumerate(zip(cases, origins.cpu().numpy())):
            centers = mesh.triangles_center
            band = terrain.terrain_origins[:, flat_cols, :].cpu().numpy().reshape(-1, 3)
            low, high = band[:, :2].min(0)-4.49999, band[:, :2].max(0)+4.49999
            keep = ((centers[:, :2] > low) & (centers[:, :2] < high)).all(1)
            tile = mesh.submesh([np.flatnonzero(keep)], append=True)
            assert len(tile.faces) > 0
            tile.apply_translation(-o)
            tile.export(output / f"terrain_{n}.obj")
            c["origin"] = o.tolist()
            c["env_id"] = int(ids[n])
        obs = self.env.get_observations().to(self.device)
        self.alg.policy.eval()
        trace = {
            k: []
            for k in (
                "q",
                "qd",
                "body_pos",
                "body_quat",
                "command",
                "velocity",
                "requested",
                "target",
                "torque",
                "force",
            )
        }
        ended = torch.zeros(len(ids), dtype=torch.bool, device=raw.device)
        ends = [None] * len(ids)
        term = raw.action_manager.get_term("joint_pos")
        with torch.inference_mode():
            for step in range(1500):
                tick[0] = step
                scripted(command)
                issued = command.commands[ids].clone()
                act = self.alg.policy.act_inference(obs)
                obs, reward, done, extra = self.env.step(act.to(raw.device))
                obs = obs.to(self.device)
                assert torch.isfinite(reward).all()
                values = dict(
                    q=robot.data.joint_pos[ids],
                    qd=robot.data.joint_vel[ids],
                    body_pos=robot.data.body_pos_w[ids] - origins[:, None, :],
                    body_quat=robot.data.body_quat_w[ids],
                    command=issued,
                    velocity=robot.data.root_lin_vel_b[ids],
                    requested=term.requested_actions[ids],
                    target=term.processed_actions[ids],
                    torque=robot.data.applied_torque[ids],
                    force=raw.scene["landing_contacts"]
                    .data.net_forces_w_history[ids]
                    .norm(dim=-1)
                    .amax(1),
                )
                for k, v in values.items():
                    trace[k].append(v.cpu().numpy().copy())
                fresh = done[ids].bool() & ~ended
                for n in fresh.nonzero().flatten().tolist():
                    names = []
                    for name in raw.termination_manager.active_terms:
                        if bool(raw.termination_manager.get_term(name)[ids[n]]):
                            names.append(name)
                    ends[n] = dict(
                        step=step, time_s=(step + 1) * raw.step_dt, terms=names
                    )
                ended |= fresh
                if bool(ended.all()):
                    break
                if step % 250 == 0:
                    print("RECORD", step, "ended", int(ended.sum()), flush=True)
        after = {
            k: state_sha256(v, torch)
            for k, v in (
                ("model", self.alg.policy.state_dict()),
                ("ppo", self.alg.optimizer.state_dict()),
                ("student", self.alg.optimizer_stu_enc.state_dict()),
            )
        }
        assert before == after
        np.savez_compressed(
            output / "traces.npz", **{k: np.stack(v) for k, v in trace.items()}
        )
        receipt = dict(
            checkpoint=str(checkpoint),
            sha256=digest(checkpoint),
            completed=self.current_learning_iteration,
            cases=cases,
            ends=ends,
            joint_names=robot.joint_names,
            body_names=robot.body_names,
            dt=raw.step_dt,
            steps=len(trace["q"]),
            default_joint_pos=q.cpu().tolist(),
            zero_optimizer_updates=True,
            state_hashes_unchanged=True,
            observation_noise="training configuration retained",
            randomization="training configuration retained",
            commands="2s zero then 28s straight 1.0 m/s; vy=yaw=0",
            interpretation="one fixed first episode per case; not a random success rate",
        )
        (output / "FINALIZED.json").write_text(json.dumps(receipt, indent=2))

    return learn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frozen()
    import early_restart

    original = early_restart.configure

    def configure(cfg, profile):
        original(cfg, profile)
        from isaaclab.terrains import TerrainImporter

        original_import = TerrainImporter.import_mesh

        def capture_mesh(importer, name, mesh):
            result = original_import(importer, name, mesh)
            if name == "terrain":
                importer._showcase_imported_mesh = mesh.copy()
            return result

        TerrainImporter.import_mesh = capture_mesh
        from rsl_rl.runners import OnPolicyRunnerCTS

        OnPolicyRunnerCTS.learn = recording(a.checkpoint, a.output)

    early_restart.configure = configure
    sys.argv = [
        str(STAGE / "code/train.py"),
        *train_args(a.checkpoint, a.output.with_name(a.output.name + "_runtime"), 0),
    ]
    from train import main as train_main

    train_main()
    assert (a.output / "FINALIZED.json").exists()


if __name__ == "__main__":
    main()
