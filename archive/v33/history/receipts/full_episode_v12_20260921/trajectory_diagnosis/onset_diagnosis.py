"""Compare V12 direct and settled forward command onset without simulation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FOCUS_JOINTS = (
    "fl_calf_joint",
    "fr_thigh_joint",
    "fl_hip_joint",
    "rl_thigh_joint",
    "rl_hip_joint",
    "rr_hip_joint",
    "fr_calf_joint",
)
POSE_RMS_LIMIT = 0.25
SINGLE_JOINT_LIMIT = 0.60


def first_crossing(values: np.ndarray, threshold: float, dt: float) -> dict[str, Any] | None:
    indices = np.argwhere(np.abs(values) > threshold)
    if not len(indices):
        return None
    time_index, column = (int(value) for value in indices[0])
    return {
        "time_s": round((time_index + 1) * dt, 4),
        "column": column,
        "value": float(values[time_index, column]),
    }


def first_scalar_crossing(values: np.ndarray, threshold: float, dt: float) -> dict[str, Any] | None:
    indices = np.flatnonzero(values > threshold)
    if not len(indices):
        return None
    index = int(indices[0])
    return {"time_s": round((index + 1) * dt, 4), "value": float(values[index])}


def peak(values: np.ndarray, dt: float) -> dict[str, Any]:
    flat = int(np.argmax(np.abs(values)))
    time_index, column = np.unravel_index(flat, values.shape)
    return {
        "time_s": round((time_index + 1) * dt, 4),
        "column": int(column),
        "value": float(values[time_index, column]),
    }


def mode_summary(
    mode: str,
    case: dict[str, Any],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case_id = case["id"]
    dt = manifest["dt"]
    names = manifest["joint_names"]
    legs = [index for index, name in enumerate(names) if not name.endswith("_foot_joint")]
    default = np.asarray(manifest["default_q"])
    request_ids = manifest["request_joint_ids"]
    request = arrays["requested"][:, case_id] - default[request_ids]
    target = arrays["target"][:, case_id][:, legs] - default[legs]
    actual = arrays["joint_pos"][:, case_id][:, legs] - default[legs]
    q_rms = np.sqrt(np.mean(actual**2, axis=1))
    issued = arrays["command"][:, case_id, 0]
    nonzero = np.flatnonzero(np.abs(issued) > 1e-6)
    if not len(nonzero):
        raise ValueError(f"no forward command in case {case_id}")
    onset_index = int(nonzero[0])
    onset_time = round((onset_index + 1) * dt, 4)

    def named(record: dict[str, Any] | None, order: list[int]) -> dict[str, Any] | None:
        if record is None:
            return None
        record = dict(record)
        record["joint"] = names[order[record.pop("column")]]
        return record

    first_second = slice(onset_index, min(len(q_rms), onset_index + round(1.0 / dt)))
    actual_peak = peak(actual[first_second], dt)
    actual_peak["time_s"] = round(actual_peak["time_s"] + onset_index * dt, 4)
    actual_peak = named(actual_peak, legs)
    peak_joint = names.index(actual_peak["joint"])
    peak_time_index = round(actual_peak["time_s"] / dt) - 1
    request_column = request_ids.index(peak_joint)

    def onset_named_crossing(
        values: np.ndarray, order: list[int], threshold: float
    ) -> dict[str, Any] | None:
        record = named(first_crossing(values[first_second], threshold, dt), order)
        if record is not None:
            record["relative_time_s"] = round(record["time_s"] - dt, 4)
            record["absolute_post_step_time_s"] = round(record["time_s"] + onset_index * dt, 4)
            del record["time_s"]
        return record

    return {
        "mode": mode,
        "case_id": case_id,
        "command_onset_trace_index": onset_index,
        "command_onset_post_step_time_s": onset_time,
        "pre_onset_control_steps_available": onset_index,
        "first_gate_crossings_from_episode_start": {
            "request_abs_gt_0.60rad": named(first_crossing(request, SINGLE_JOINT_LIMIT, dt), request_ids),
            "guarded_target_pre_actuator_delay_abs_gt_0.60rad": named(
                first_crossing(target, SINGLE_JOINT_LIMIT, dt), legs
            ),
            "actual_abs_gt_0.60rad": named(first_crossing(actual, SINGLE_JOINT_LIMIT, dt), legs),
            "actual_qrms_gt_0.25rad": first_scalar_crossing(q_rms, POSE_RMS_LIMIT, dt),
        },
        "first_crossings_after_command_onset": {
            "request_abs_gt_0.60rad": onset_named_crossing(
                request, request_ids, SINGLE_JOINT_LIMIT
            ),
            "guarded_target_pre_actuator_delay_abs_gt_0.60rad": onset_named_crossing(
                target, legs, SINGLE_JOINT_LIMIT
            ),
            "actual_abs_gt_0.60rad": onset_named_crossing(actual, legs, SINGLE_JOINT_LIMIT),
        },
        "first_second_max_actual_from_default": actual_peak,
        "max_joint_snapshot": {
            "joint": names[peak_joint],
            "time_s": actual_peak["time_s"],
            "request_from_default_rad": float(request[peak_time_index, request_column]),
            "guarded_target_pre_actuator_delay_from_default_rad": float(
                target[peak_time_index, legs.index(peak_joint)]
            ),
            "actual_from_default_rad": float(actual[peak_time_index, legs.index(peak_joint)]),
            "actual_minus_guarded_target_rad": float(
                actual[peak_time_index, legs.index(peak_joint)]
                - target[peak_time_index, legs.index(peak_joint)]
            ),
            "joint_velocity_rad_s": float(arrays["joint_vel"][peak_time_index, case_id, peak_joint]),
            "control_step_speed_peak_rad_s": float(arrays["speed_peak"][peak_time_index, case_id, peak_joint]),
            "control_step_torque_peak_nm": float(arrays["torque_peak"][peak_time_index, case_id, peak_joint]),
        },
        "first_second_qrms_peak": float(q_rms[first_second].max()),
    }


def write_control_csv(
    path: Path,
    cases: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    names = manifest["joint_names"]
    default = np.asarray(manifest["default_q"])
    request_ids = manifest["request_joint_ids"]
    request_column = {joint_id: column for column, joint_id in enumerate(request_ids)}
    wheel_names = [name for name in names if name.endswith("_foot_joint")]
    rows = []
    for mode, case in cases.items():
        case_id = case["id"]
        issued = arrays["command"][:, case_id, 0]
        onset = int(np.flatnonzero(np.abs(issued) > 1e-6)[0])
        left = max(0, onset - 10)
        right = min(len(issued), onset + 11)
        for index in range(left, right):
            action = arrays["raw_action"][index, case_id]
            common = {
                "mode": mode,
                "case_id": case_id,
                "trace_index": index,
                "post_step_time_s": round((index + 1) * manifest["dt"], 4),
                "relative_to_first_nonzero_issued_command_s": round((index - onset) * manifest["dt"], 4),
                "issued_command_vx_m_s": float(issued[index]),
                "actual_base_vx_m_s": float(arrays["velocity"][index, case_id, 0]),
                "policy_action_signed_mean_all16": float(action.mean()),
                "policy_action_abs_mean_all16": float(np.abs(action).mean()),
                "policy_action_abs_mean_legs12": float(np.abs(action[:12]).mean()),
                "policy_action_abs_mean_wheels4": float(np.abs(action[12:]).mean()),
                "base_contact_peak_n": float(arrays["base_contact"][index, case_id]),
                "wheel_contact_max_n": float(arrays["wheel_ground_force"][index, case_id].max()),
            }
            for wheel_index, wheel_name in enumerate(wheel_names):
                common[f"{wheel_name}_contact_n"] = float(
                    arrays["wheel_ground_force"][index, case_id, wheel_index]
                )
            for name in FOCUS_JOINTS:
                joint_id = names.index(name)
                column = request_column[joint_id]
                rows.append(
                    {
                        **common,
                        "joint": name,
                        "policy_raw_action": float(action[column]),
                        "original_request_rad": float(arrays["requested"][index, case_id, column]),
                        "request_from_default_rad": float(
                            arrays["requested"][index, case_id, column] - default[joint_id]
                        ),
                        "guarded_target_pre_actuator_delay_rad": float(
                            arrays["target"][index, case_id, joint_id]
                        ),
                        "guarded_target_pre_actuator_delay_from_default_rad": float(
                            arrays["target"][index, case_id, joint_id] - default[joint_id]
                        ),
                        "actual_rad": float(arrays["joint_pos"][index, case_id, joint_id]),
                        "actual_from_default_rad": float(
                            arrays["joint_pos"][index, case_id, joint_id] - default[joint_id]
                        ),
                        "actual_minus_guarded_target_rad": float(
                            arrays["joint_pos"][index, case_id, joint_id]
                            - arrays["target"][index, case_id, joint_id]
                        ),
                        "joint_velocity_rad_s": float(arrays["joint_vel"][index, case_id, joint_id]),
                        "control_step_speed_peak_rad_s": float(
                            arrays["speed_peak"][index, case_id, joint_id]
                        ),
                        "control_step_torque_peak_nm": float(
                            arrays["torque_peak"][index, case_id, joint_id]
                        ),
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_key_steps_csv(
    path: Path,
    cases: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    names = manifest["joint_names"]
    default = np.asarray(manifest["default_q"])
    request_ids = manifest["request_joint_ids"]
    legs = [index for index, name in enumerate(names) if not name.endswith("_foot_joint")]
    rows = []
    for mode, case in cases.items():
        case_id = case["id"]
        issued = arrays["command"][:, case_id, 0]
        onset = int(np.flatnonzero(np.abs(issued) > 1e-6)[0])
        offsets = (0, 1, 5, 10) if mode == "direct" else (-10, -5, -1, 0, 1, 5, 10)
        for offset in offsets:
            index = onset + offset
            action = arrays["raw_action"][index, case_id]
            request = arrays["requested"][index, case_id] - default[request_ids]
            target = arrays["target"][index, case_id, legs] - default[legs]
            actual = arrays["joint_pos"][index, case_id, legs] - default[legs]
            torque = arrays["torque_peak"][index, case_id, legs]
            speed = arrays["speed_peak"][index, case_id, legs]

            def extreme(values: np.ndarray, order: list[int]) -> tuple[str, float]:
                column = int(np.argmax(np.abs(values)))
                return names[order[column]], float(values[column])

            request_joint, request_value = extreme(request, request_ids)
            target_joint, target_value = extreme(target, legs)
            actual_joint, actual_value = extreme(actual, legs)
            torque_joint, torque_value = extreme(torque, legs)
            speed_joint, speed_value = extreme(speed, legs)
            rows.append(
                {
                    "mode": mode,
                    "relative_to_onset_s": round(offset * manifest["dt"], 4),
                    "post_step_time_s": round((index + 1) * manifest["dt"], 4),
                    "issued_command_vx_m_s": float(issued[index]),
                    "actual_base_vx_m_s": float(arrays["velocity"][index, case_id, 0]),
                    "policy_action_signed_mean_all16": float(action.mean()),
                    "policy_action_abs_mean_all16": float(np.abs(action).mean()),
                    "q_rms_rad": float(np.sqrt(np.mean(actual**2))),
                    "max_request_joint": request_joint,
                    "max_request_from_default_rad": request_value,
                    "max_guarded_target_pre_actuator_delay_joint": target_joint,
                    "max_guarded_target_pre_actuator_delay_from_default_rad": target_value,
                    "max_actual_joint": actual_joint,
                    "max_actual_from_default_rad": actual_value,
                    "max_torque_joint": torque_joint,
                    "max_torque_nm": torque_value,
                    "max_speed_joint": speed_joint,
                    "max_speed_rad_s": speed_value,
                    "wheel_contact_max_n": float(
                        arrays["wheel_ground_force"][index, case_id].max()
                    ),
                    "base_contact_peak_n": float(arrays["base_contact"][index, case_id]),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def appendix(summary: dict[str, Any]) -> str:
    direct = summary["cases"]["direct"]
    settle = summary["cases"]["settle"]
    d_cross = direct["first_gate_crossings_from_episode_start"]
    d_peak = direct["max_joint_snapshot"]
    s_peak = settle["max_joint_snapshot"]
    return f"""# Direct 与 settle 前进命令切入附录

## 结论

V12 direct 前进的 **1.0284rad** 最大单关节偏移来自 `fl_calf_joint`，发生在 **0.58s**，并非此前重点查看的 `rl_thigh_joint`。该时刻原请求为 {d_peak['request_from_default_rad']:.4f}rad，guard 后、actuator delay 前的 `data.joint_pos_target` 为 {d_peak['guarded_target_pre_actuator_delay_from_default_rad']:.4f}rad，actual 为 {d_peak['actual_from_default_rad']:.4f}rad，两者差 {d_peak['actual_minus_guarded_target_rad']:.4f}rad。这个峰与 guard 后 target 贴近，支持“策略请求把受限目标持续推到 calf 下限附近”；但收据未保存 delay buffer 实际送入 PD 的位置目标，所以不能把该差值严格解释成执行器跟踪误差，也不能仅凭本 trace 完全排除 delay 层内的瞬态。

最早时序也支持这个区分：direct 的 request 首次超过 0.60rad 在 **{d_cross['request_abs_gt_0.60rad']['time_s']:.2f}s**（`{d_cross['request_abs_gt_0.60rad']['joint']}`），guard 后 target 首次超过 0.60rad 同样在 **{d_cross['guarded_target_pre_actuator_delay_abs_gt_0.60rad']['time_s']:.2f}s**，actual qRMS 在 **{d_cross['actual_qrms_gt_0.25rad']['time_s']:.2f}s** 越过 0.25rad，单关节 actual 到 **{d_cross['actual_abs_gt_0.60rad']['time_s']:.2f}s** 才超过 0.60rad。0.02s 的第一个控制步已同时观察到非默认 request/target、120Nm 转矩峰和 15rad/s 速度峰；50Hz post-step request 与控制步内 200Hz 峰无法继续判断这 20ms 内谁先发生。后续可分辨时序是 request/guarded target 先越门、actual 后越门，因此证据优先指向策略请求与受限目标序列；对未记录 delay 后目标的执行器瞬态保持保留。

settle 并不表示网络原请求始终很小：命令切入前 0.20s，`rl_hip_joint` request 相对默认姿态已约 **-4.7740rad**；切入第一步仍为 **-4.6284rad**，但 guard 后 target 保持约 **-0.1690rad**，actual 约 **-0.0159rad**。这项大 request 在零命令历史中已经存在，并非 0.6m/s 边沿新产生。settle 首秒最大 actual 偏移是 `{s_peak['joint']}` 的 **{abs(s_peak['actual_from_default_rad']):.4f}rad@{s_peak['time_s']:.2f}s**。所以 request 绝对幅值或 mean policy action 都不能单独解释 direct 拒门；关键观测是 direct 的多关节请求经过 guard 后仍形成了能把 actual 推过姿态门的目标序列。

## 命令边沿关键步

|模式|相对切入|qRMS|mean abs policy action|最大 request|最大 guard 后 target|最大 actual|腿转矩/速度峰|最大轮接触|
|---|---:|---:|---:|---|---|---|---:|---:|
|direct|0.00s|0.0897|0.9780|fl_hip -0.5927|fr_thigh +0.2400|fr_calf -0.1439|120.0Nm / 15.0000rad/s|0.0N|
|direct|+0.10s|0.1985|1.8084|fl_hip -0.7694|fr_thigh +0.6416|fr_thigh +0.3760|32.1Nm / 6.8124rad/s|200.5N|
|direct|+0.20s|0.4295|1.7677|fl_calf -1.1308|fl_calf -1.0230|fl_calf -0.7956|30.8Nm / 4.4025rad/s|231.2N|
|settle|-0.20s|0.0512|2.7065|rl_hip -4.7740|fl_hip +0.1690|fr_calf -0.1134|21.3Nm / 0.2677rad/s|123.4N|
|settle|0.00s|0.0519|2.9268|rl_hip -4.6284|fl_thigh -0.2731|fr_calf -0.1178|35.3Nm / 0.8174rad/s|228.2N|
|settle|+0.20s|0.0907|1.9328|rl_hip -2.5555|fl_hip +0.1690|fr_calf -0.2144|21.6Nm / 1.3071rad/s|129.6N|

direct 在 +0.20s 已达到 qRMS 0.4295，而 settle 在相同相对时刻仍为 0.0907。反过来，settle 的 mean abs action 与最大原请求更大，却被 guard 隔离在较小 target/actual 范围内。这再次说明不能按 raw action 或 request 单值排序风险，必须看 request→guarded target→actual 的整条时序。

## 两个 case 的真实差别

- evaluator 对两者使用同一 V12 student checkpoint、同一 seed、同一 flat 几何、同一默认关节/根状态、关闭 observation corruption 和其它 events。两者 case 定义只在 `mode` 不同。
- direct 从 reset 后第一帧就给 `vx=0.6m/s`；settle 在前 2 秒给零命令，随后一步切到相同 `vx=0.6m/s`。CSV 中 settle 有命令切入前 0.2 秒数据；direct 的命令在首个 policy observation 前已经非零，因此不存在同一 episode 内的负时间控制步，只有 manifest 的初始状态可作 t=0 参考。
- policy observation 是 10 帧历史，每帧 53 维。direct 使用 reset 起点的历史立即动作；settle 在命令切入时已有约 100 个站立控制步，输入的最近 10 帧覆盖约 0.2 秒零命令站立状态。这是两种协议除命令时刻外最重要的输入历史差异。
- evaluator 实例化的是 `Go2Env`，不是带随机子步 action delay 的 `ActionDelayGo2Env`。所有 actuator 的 delay 固定为 2 个 5ms physics steps，即 10ms；direct 与 settle 相同。
- 训练的 `omni_full_episode_v1` 保留 `normal_reset_root/joints` 和普通 5 秒 command sampler；但 evaluator 在配置 profile 后覆盖 reset 函数，强制默认关节、零速度和零 pose/velocity range，再用脚本命令。这里的 direct/settle 是 matched fixed-case 诊断，settle 不是训练 sampler 的独立随机分支。

## 控制步文件

- `onset_control_steps.csv`：两 case 命令切入前后 0.2 秒；主要关节的 policy raw action、原请求、guard 后/delay 前 target、actual、误差、关节速度、控制步内 200Hz 转矩/速度峰，以及全动作均值、base 与四轮接触。
- `onset_key_steps.csv`：关键相对时刻的宽表，直接给出全腿最大 request/target/actual、qRMS、动作均值、转矩、速度和轮接触。
- `onset_summary.json`：门限首次越界、最大单关节快照、输入 case 与源码契约。

注意：trace 中的 `target` 是 `robot.data.joint_pos_target`。本次实际 remote runtime 的 `/root/autodl-tmp/workspace/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` 已核对：第 1101 行写 `data.joint_pos_target`，第 1880 行把该字段组装进 control action，第 1886 行才调用 `actuator.compute()`，第 1893 行仅在返回位置非空时写独立 sim target。线上同版本 [`articulation.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py#L1490-L1497) 也把 data target 定义为“sent to the actuator from the user”。[`DelayedPDActuator.compute`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py#L323-L331) 先用 `positions_delay_buffer.compute()` 产生送入 PD 的延迟位置目标，而显式 [`IdealPDActuator.compute`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py#L167-L181) 随后输出 effort 并清空 `joint_positions`。因此对本次 Thunder 显式电机，`_joint_pos_target_sim` 不是可预设为有效的 delay 后位置目标；本收据能区分 original request、guard 后/delay 前 target 和 actual，**没有**记录 delay buffer 实际送入 PD 的位置目标。

## 下一单因素待验假设

保持初始状态、策略、固定 10ms delay 和 `vx=0.6m/s` 不变，只改变命令边沿，比较 t=0 step 与 0.2/0.5/1.0 秒线性 ramp；同时新增 guard 前 request、guard 后/delay 前 target，以及 `positions_delay_buffer.compute()` 返回并送入 PD 的位置目标，配套记录 computed/applied effort。另可在**离线网络输入反事实**中替换历史帧，判断 actor 输出对历史的敏感性，但复制 settle 历史会脱离 reset 时真实物理状态，不能作为 on-policy 行为能力或 gate 验收。该建议是待验假设，不是现有数据的因果结论。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.receipt / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((args.receipt / "analysis.json").read_text(encoding="utf-8"))
    cases = {
        mode: next(
            case
            for case in analysis["cases"]
            if case["label"] == "V12_45011"
            and case["direction"] == "forward"
            and case["mode"] == mode
        )
        for mode in ("direct", "settle")
    }
    names = (
        "joint_pos",
        "joint_vel",
        "target",
        "requested",
        "raw_action",
        "command",
        "velocity",
        "torque_peak",
        "speed_peak",
        "base_contact",
        "wheel_ground_force",
    )
    with np.load(args.receipt / "traces.npz") as archive:
        arrays = {name: archive[name] for name in names}
    summary = {
        "scope": "read-only analysis of two fixed V12 student forward cases",
        "trace_contract": {
            "dt_s": manifest["dt"],
            "policy_history_frames": 10,
            "single_observation_width": 53,
            "actuator_delay_physics_steps": 2,
            "physics_dt_s": 0.005,
            "actuator_delay_s": 0.010,
            "target_semantics": "robot.data.joint_pos_target: post-guard, pre-actuator-delay user command",
        },
        "focus_joints": list(FOCUS_JOINTS),
        "cases": {
            mode: mode_summary(mode, case, arrays, manifest) for mode, case in cases.items()
        },
    }
    output = Path(__file__).parent
    write_control_csv(output / "onset_control_steps.csv", cases, arrays, manifest)
    write_key_steps_csv(output / "onset_key_steps.csv", cases, arrays, manifest)
    (output / "onset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "APPENDIX_ONSET.md").write_text(appendix(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
