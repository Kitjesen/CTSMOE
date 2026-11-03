"""Independent trajectory diagnosis for the fixed V12 pilot-300 evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PIT_HEIGHT_M = {0: 0.15, 2: 0.25, 4: 0.35, 6: 0.45, 9: 0.60, 15: 0.90, 21: 1.20}
KEY_JOINTS = ("rl_thigh_joint", "rr_hip_joint", "fr_calf_joint")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def window_bounds(start_s: float, stop_s: float, dt: float, count: int) -> tuple[int, int]:
    left = min(count, max(0, int(round(start_s / dt))))
    right = min(count, max(left, int(round(stop_s / dt))))
    return left, right


def peak(values: np.ndarray, dt: float, *, absolute: bool = True) -> dict[str, Any]:
    measured = np.abs(values) if absolute else values
    flat_index = int(np.nanargmax(measured))
    index = np.unravel_index(flat_index, measured.shape)
    record: dict[str, Any] = {
        "value": finite_float(measured[index]),
        "time_s": round((index[0] + 1) * dt, 4),
    }
    if measured.ndim > 1:
        record["column"] = int(index[1])
    return record


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["terrain"],
        case["level"],
        case["mode"],
        case["direction"],
        tuple(case["velocity"]),
    )


def case_brief(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "label",
        "terrain",
        "level",
        "mode",
        "direction",
        "velocity",
        "seconds",
        "pass",
        "physical",
        "terms",
        "straight_4m",
        "dx_m",
        "climb",
        "stable",
    )
    return {key: case.get(key) for key in keys if key in case}


def comparison_pairs(cases: list[dict[str, Any]], old_label: str, new_label: str) -> dict[str, Any]:
    old = {case_key(case): case for case in cases if case["label"] == old_label}
    new = {case_key(case): case for case in cases if case["label"] == new_label}
    if old.keys() != new.keys():
        raise ValueError(f"unmatched cases for {old_label} and {new_label}")
    gained = []
    lost = []
    for key in sorted(old, key=str):
        before, after = old[key], new[key]
        item = {"before": case_brief(before), "after": case_brief(after)}
        if after["pass"] and not before["pass"]:
            gained.append(item)
        elif before["pass"] and not after["pass"]:
            lost.append(item)
    return {
        "baseline": old_label,
        "candidate": new_label,
        "matched_cases": len(old),
        "gained": gained,
        "lost": lost,
    }


def matrix(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["label"], case["terrain"])].append(case)
    rows = []
    for (label, terrain), items in sorted(grouped.items()):
        rows.append(
            {
                "label": label,
                "terrain": terrain,
                "count": len(items),
                "pass": sum(bool(item["pass"]) for item in items),
                "physics": sum(bool(item["physical"]) for item in items),
                "straight": sum(bool(item.get("straight_4m", False)) for item in items),
                "term_counts": dict(
                    sorted(
                        {
                            term: sum(term in item["terms"] for item in items)
                            for term in {term for item in items for term in item["terms"]}
                        }.items()
                    )
                ),
            }
        )
    return rows


def trajectory_record(
    case: dict[str, Any], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> dict[str, Any]:
    case_id = case["id"]
    count = case["end_step"] + 1
    dt = manifest["dt"]
    joint_names = manifest["joint_names"]
    legs = [index for index, name in enumerate(joint_names) if not name.endswith("_foot_joint")]
    default = np.asarray(manifest["default_q"])[legs]
    q_error = arrays["joint_pos"][:count, case_id][:, legs] - default
    q_rms = np.sqrt(np.mean(q_error**2, axis=1))
    velocity_error = arrays["velocity"][:count, case_id] - arrays["command"][:count, case_id]
    height = arrays["root_clearance"][:count, case_id]
    torque = arrays["torque_peak"][:count, case_id]
    speed = arrays["speed_peak"][:count, case_id]
    contact = arrays["base_contact"][:count, case_id]
    onset = 2.0 if case["mode"] == "settle" else 0.0
    early = window_bounds(onset, onset + 1.0, dt, count)
    late = window_bounds(max(onset, case["seconds"] - 2.0), case["seconds"], dt, count)

    def metrics(bounds: tuple[int, int]) -> dict[str, Any] | None:
        left, right = bounds
        if right <= left:
            return None
        return {
            "samples": right - left,
            "speed_error_xy_mean": finite_float(np.linalg.norm(velocity_error[left:right, :2], axis=1).mean()),
            "speed_actual_xy_mean": finite_float(np.linalg.norm(arrays["velocity"][left:right, case_id, :2], axis=1).mean()),
            "height_mean_m": finite_float(height[left:right].mean()),
            "height_range_m": finite_float(np.ptp(height[left:right])),
            "q_rms_mean": finite_float(q_rms[left:right].mean()),
            "q_rms_peak": finite_float(q_rms[left:right].max()),
            "leg_torque_peak_nm": finite_float(torque[left:right][:, legs].max()),
            "leg_speed_peak_rad_s": finite_float(speed[left:right][:, legs].max()),
            "base_contact_peak_n": finite_float(contact[left:right].max()),
        }

    torque_peak = peak(torque[:, legs], dt)
    torque_peak["joint"] = joint_names[legs[torque_peak.pop("column")]]
    speed_peak = peak(speed[:, legs], dt)
    speed_peak["joint"] = joint_names[legs[speed_peak.pop("column")]]
    q_peak = peak(q_rms, dt)
    height_low = peak(-height, dt, absolute=False)
    height_low["value"] = finite_float(-height_low["value"])
    return {
        **case_brief(case),
        "early_after_command": metrics(early),
        "last_2s": metrics(late),
        "q_rms_peak": q_peak,
        "height_min": height_low,
        "leg_torque_peak": torque_peak,
        "leg_speed_peak": speed_peak,
        "base_contact_peak": peak(contact, dt),
    }


def pit_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pit = [case for case in cases if case["terrain"] == "pit"]
    unknown = sorted({case["level"] for case in pit} - PIT_HEIGHT_M.keys())
    if unknown:
        raise ValueError(f"unknown pit levels: {unknown}")
    rows = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in pit:
        grouped[(case["label"], case["level"])].append(case)
    for (label, level), items in sorted(grouped.items()):
        rows.append(
            {
                "label": label,
                "level": level,
                "actual_height_m": PIT_HEIGHT_M[level],
                "count": len(items),
                "pass": sum(item["pass"] for item in items),
                "physics": sum(item["physical"] for item in items),
                "successful_speeds_m_s": [item["velocity"][0] for item in items if item["pass"]],
                "failed_speeds_m_s": [item["velocity"][0] for item in items if not item["pass"]],
            }
        )
    return {"by_height": rows, "failures": [case_brief(case) for case in pit if not case["pass"]]}


def stair_jitter(
    cases: list[dict[str, Any]], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    dt = manifest["dt"]
    joint_names = manifest["joint_names"]
    legs = [index for index, name in enumerate(joint_names) if not name.endswith("_foot_joint")]
    for case in cases:
        if not case["terrain"].startswith("stairs"):
            continue
        case_id = case["id"]
        count = case["end_step"] + 1
        left = max(0, count - round(2.0 / dt))
        velocity = arrays["velocity"][left:count, case_id]
        command = arrays["command"][left:count, case_id]
        height = arrays["root_clearance"][left:count, case_id]
        records.append(
            {
                **case_brief(case),
                "last_2s_samples": count - left,
                "forward_velocity_std": finite_float(velocity[:, 0].std()),
                "forward_velocity_peak_to_peak": finite_float(np.ptp(velocity[:, 0])),
                "xy_tracking_error_mean": finite_float(np.linalg.norm(velocity[:, :2] - command[:, :2], axis=1).mean()),
                "height_std_m": finite_float(height.std()),
                "height_peak_to_peak_m": finite_float(np.ptp(height)),
                "leg_torque_peak_nm": finite_float(arrays["torque_peak"][left:count, case_id][:, legs].max()),
                "leg_speed_peak_rad_s": finite_float(arrays["speed_peak"][left:count, case_id][:, legs].max()),
                "base_contact_peak_n": finite_float(arrays["base_contact"][left:count, case_id].max()),
                "reported_end_vz_peak": case.get("end_vz_peak"),
            }
        )
    return records


def forward_reverse_trajectories(
    cases: list[dict[str, Any]], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        trajectory_record(case, arrays, manifest)
        for case in cases
        if case["terrain"] == "flat" and case["direction"] in ("forward", "reverse")
    ]


def joint_timing(
    case: dict[str, Any], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    count = case["end_step"] + 1
    dt = manifest["dt"]
    names = manifest["joint_names"]
    default = np.asarray(manifest["default_q"])
    request_joint_ids = manifest["request_joint_ids"]
    request_column = {joint_id: column for column, joint_id in enumerate(request_joint_ids)}
    records = []
    for joint_id, name in enumerate(names):
        if joint_id not in request_column:
            continue
        actual = arrays["joint_pos"][:count, case["id"], joint_id]
        target = arrays["target"][:count, case["id"], joint_id]
        request = arrays["requested"][:count, case["id"], request_column[joint_id]]
        actual_peak = peak(actual - default[joint_id], dt)
        target_peak = peak(target - default[joint_id], dt)
        request_peak = peak(request - default[joint_id], dt)
        tracking_peak = peak(actual - target, dt)
        request_target_peak = peak(request - target, dt)
        records.append(
            {
                "joint": name,
                "emphasized": name in KEY_JOINTS,
                "actual_from_default_peak_rad": actual_peak,
                "target_from_default_peak_rad": target_peak,
                "request_from_default_peak_rad": request_peak,
                "actual_target_error_peak_rad": tracking_peak,
                "request_target_gap_peak_rad": request_target_peak,
            }
        )
    records.sort(
        key=lambda item: item["actual_target_error_peak_rad"]["value"] or -1.0,
        reverse=True,
    )
    return records


def flat25_forward(
    cases: list[dict[str, Any]], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    dt = manifest["dt"]
    names = manifest["joint_names"]
    legs = [index for index, name in enumerate(names) if not name.endswith("_foot_joint")]
    default = np.asarray(manifest["default_q"])[legs]
    result = []
    for case in cases:
        if case["terrain"] != "flat" or case["direction"] != "forward":
            continue
        count = case["end_step"] + 1
        q = arrays["joint_pos"][:count, case["id"]][:, legs] - default
        q_rms = np.sqrt(np.mean(q**2, axis=1))
        onset = 2.0 if case["mode"] == "settle" else 0.0

        def rms_window(start: float, stop: float) -> dict[str, Any] | None:
            left, right = window_bounds(start, stop, dt, count)
            if right <= left:
                return None
            values = q_rms[left:right]
            at = int(np.argmax(values))
            return {
                "window_s": [start, stop],
                "mean": finite_float(values.mean()),
                "peak": finite_float(values[at]),
                "peak_time_s": round((left + at + 1) * dt, 4),
            }

        result.append(
            {
                **case_brief(case),
                "completed_25s": case.get("completed_25s"),
                "startup_0_1s": rms_window(0.0, 1.0),
                "first_second_after_command": rms_window(onset, onset + 1.0),
                "late_20_25s": rms_window(20.0, 25.0),
                "trajectory": trajectory_record(case, arrays, manifest),
                "joints_ranked_by_actual_target_error": joint_timing(case, arrays, manifest),
            }
        )
    return result


def load_arrays(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(directory / "traces.npz") as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"missing trace arrays in {directory}: {sorted(missing)}")
        return {name: archive[name] for name in names}


TRACE_NAMES = (
    "joint_pos",
    "target",
    "requested",
    "velocity",
    "command",
    "root_clearance",
    "torque_peak",
    "speed_peak",
    "base_contact",
)


def diagnose_normal(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    cases = read_json(directory / "analysis.json")["cases"]
    arrays = load_arrays(directory, TRACE_NAMES)
    failures = [case for case in cases if not case["pass"]]
    result = {
        "protocol": manifest["protocol"],
        "case_count": len(cases),
        "matrix": matrix(cases),
        "student_comparison": comparison_pairs(cases, "V9_44711", "V12_45011"),
        "teacher_comparison": comparison_pairs(cases, "V9_44711_teacher", "V12_45011_teacher"),
        "pit": pit_summary(cases),
        "flat_forward_reverse_trajectories": forward_reverse_trajectories(cases, arrays, manifest),
        "stair_last_2s_jitter": stair_jitter(cases, arrays, manifest),
        "failure_trajectories": [trajectory_record(case, arrays, manifest) for case in failures],
    }
    return result


def diagnose_flat25(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    cases = read_json(directory / "analysis.json")["cases"]
    arrays = load_arrays(directory, TRACE_NAMES)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["label"], case["mode"])].append(case)
    completion = [
        {
            "label": label,
            "mode": mode,
            "count": len(items),
            "completed_25s": sum(bool(item.get("completed_25s")) for item in items),
            "pass": sum(bool(item["pass"]) for item in items),
            "physics": sum(bool(item["physical"]) for item in items),
        }
        for (label, mode), items in sorted(grouped.items())
    ]
    return {
        "protocol": manifest["protocol"],
        "case_count": len(cases),
        "completion_matrix": completion,
        "candidate_vs_v9_student": comparison_pairs(cases, "V9_44711", "V12_45011"),
        "candidate_vs_v9_teacher": comparison_pairs(cases, "V9_44711_teacher", "V12_45011_teacher"),
        "forward_qrms_and_joint_timing": flat25_forward(cases, arrays, manifest),
        "forward_reverse_trajectories": forward_reverse_trajectories(cases, arrays, manifest),
        "failures": [case_brief(case) for case in cases if not case["pass"]],
    }


def markdown(report: dict[str, Any]) -> str:
    normal = report["normal"]
    flat = report["flat25"]
    total = {}
    for row in normal["matrix"]:
        item = total.setdefault(row["label"], {"pass": 0, "count": 0, "physics": 0, "straight": 0})
        for key in item:
            item[key] += row[key]
    flat_total = defaultdict(lambda: {"pass": 0, "count": 0, "physics": 0, "completed": 0})
    for row in flat["completion_matrix"]:
        item = flat_total[row["label"]]
        item["pass"] += row["pass"]
        item["count"] += row["count"]
        item["physics"] += row["physics"]
        item["completed"] += row["completed_25s"]
    forward = {
        (item["label"], item["mode"]): item for item in flat["forward_qrms_and_joint_timing"]
    }
    stair_worst = max(
        normal["stair_last_2s_jitter"], key=lambda item: item["forward_velocity_peak_to_peak"]
    )
    lines = [
        "# V12 pilot300 轨迹诊断",
        "",
        "## 口径与结论",
        "",
        "本报告分析的是协议预先枚举的 fixed cases，不是随机重复试验，因此 `pass/count` 仅表示固定 case 覆盖，不解释为成功概率。V6 student 只出现在 flat25 参考中。",
        "",
        f"normal 共 {normal['case_count']} 个 case；flat25 共 {flat['case_count']} 个 case。原始输入哈希与逐 case 数据在 `diagnosis.json`。",
        "",
        "## 主要发现",
        "",
        f"- 九地形总计：V12 student {total['V12_45011']['pass']}/{total['V12_45011']['count']}，V9 student "
        f"{total['V9_44711']['pass']}/{total['V9_44711']['count']}；逐 case 实际为 "
        f"{len(normal['student_comparison']['gained'])} 个 gain、{len(normal['student_comparison']['lost'])} 个 loss。"
        f"V12 teacher 为 {total['V12_45011_teacher']['pass']}/{total['V12_45011_teacher']['count']}，"
        f"相对 V9 teacher 是 {len(normal['teacher_comparison']['gained'])} gain、{len(normal['teacher_comparison']['lost'])} loss。",
        f"- flat25：所有策略的 {sum(item['completed'] for item in flat_total.values())}/{sum(item['count'] for item in flat_total.values())} 个 case 都完成 25s，"
        f"无 physics termination。V12 student {flat_total['V12_45011']['pass']}/20，V9 student "
        f"{flat_total['V9_44711']['pass']}/20，V6 student 参考 {flat_total['V6_43202']['pass']}/20；"
        f"V12 teacher {flat_total['V12_45011_teacher']['pass']}/20。",
        f"- V12 student 直接前进的 0–1s qRMS 峰为 {forward[('V12_45011', 'direct')]['startup_0_1s']['peak']:.4f}rad@"
        f"{forward[('V12_45011', 'direct')]['startup_0_1s']['peak_time_s']:.2f}s，20–25s 峰降为 "
        f"{forward[('V12_45011', 'direct')]['late_20_25s']['peak']:.4f}rad；settle 前进命令后首秒峰仅 "
        f"{forward[('V12_45011', 'settle')]['first_second_after_command']['peak']:.4f}rad，并通过 gate。",
        "- 高台：V12 student 在 0.25/0.35/0.45/0.60m 的三个固定速度均通过；0.15m 的 0.6m/s 从 V9 pass 变为 "
        "`pit_outside`，0.90m 与 1.20m 三个速度均未通过。",
        f"- 楼梯末段最大前向速度峰峰值出现在 {stair_worst['label']} {stair_worst['terrain']} L{stair_worst['level']} "
        f"{stair_worst['velocity'][0]}m/s：{stair_worst['forward_velocity_peak_to_peak']:.4f}m/s；该 case "
        f"pass={stair_worst['pass']}，terms={stair_worst['terms']}，末段腿关节速度峰 "
        f"{stair_worst['leg_speed_peak_rad_s']:.4f}rad/s。",
        "",
        "## 九地形矩阵",
        "",
        "|策略|地形|pass/count|physics|straight|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in normal["matrix"]:
        lines.append(
            f"|{row['label']}|{row['terrain']}|{row['pass']}/{row['count']}|{row['physics']}|{row['straight']}|"
        )
    for title, key in (("学生 V12 对 V9", "student_comparison"), ("教师 V12 对 V9", "teacher_comparison")):
        item = normal[key]
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                f"逐 case 配对 {item['matched_cases']} 个：新增通过 {len(item['gained'])}，失去通过 {len(item['lost'])}。",
            ]
        )
        for kind, chinese in (("lost", "失去"), ("gained", "新增")):
            if item[kind]:
                lines.append(f"{chinese} case：")
                for pair in item[kind]:
                    case = pair["after"]
                    lines.append(
                        f"- {case['terrain']} L{case['level']} {case['direction']} {case['velocity']}，"
                        f"V12 terms={case['terms']} physical={case['physical']}。"
                    )
    lines.extend(["", "## 高台实际高度", "", "|策略|高度 m|pass/count|physics|成功速度|失败速度|", "|---|---:|---:|---:|---|---|"])
    for row in normal["pit"]["by_height"]:
        lines.append(
            f"|{row['label']}|{row['actual_height_m']:.2f}|{row['pass']}/{row['count']}|{row['physics']}|"
            f"{row['successful_speeds_m_s']}|{row['failed_speeds_m_s']}|"
        )
    lines.extend(
        [
            "",
            "## flat25 完整性",
            "",
            "|策略|模式|完成25s/count|pass|physics|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in flat["completion_matrix"]:
        lines.append(
            f"|{row['label']}|{row['mode']}|{row['completed_25s']}/{row['count']}|{row['pass']}|{row['physics']}|"
        )
    lines.extend(
        [
            "",
            "## flat25 前进 qRMS",
            "",
            "首段为 0–1s；settle 的命令在 2s 开始，因此另列命令后的首秒；晚期为 20–25s。",
            "",
            "|策略|模式|pass|0–1s peak@t|命令后首秒 peak@t|20–25s peak@t|",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in flat["forward_qrms_and_joint_timing"]:
        startup = row["startup_0_1s"]
        onset = row["first_second_after_command"]
        late = row["late_20_25s"]
        lines.append(
            f"|{row['label']}|{row['mode']}|{row['pass']}|{startup['peak']:.4f}@{startup['peak_time_s']:.2f}s|"
            f"{onset['peak']:.4f}@{onset['peak_time_s']:.2f}s|{late['peak']:.4f}@{late['peak_time_s']:.2f}s|"
        )
    lines.extend(
        [
            "",
            "## V12 student 前进/后退早期与末段",
            "",
            "早期指命令生效后的首秒，末段指 23–25s。转矩与关节速度均为腿部峰值；接触为 base contact 峰。",
            "",
            "|模式/方向|区间|实际XY速度均值|高度均值/范围 m|转矩 Nm|关节速度 rad/s|base接触 N|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in flat["forward_reverse_trajectories"]:
        if row["label"] != "V12_45011":
            continue
        for interval, key in (("early", "early_after_command"), ("late", "last_2s")):
            item = row[key]
            lines.append(
                f"|{row['mode']}/{row['direction']}|{interval}|{item['speed_actual_xy_mean']:.4f}|"
                f"{item['height_mean_m']:.4f}/{item['height_range_m']:.4f}|{item['leg_torque_peak_nm']:.4f}|"
                f"{item['leg_speed_peak_rad_s']:.4f}|{item['base_contact_peak_n']:.1f}|"
            )
    lines.extend(
        [
            "",
            "## 关节异常时间与具体失败轨迹",
            "",
            "`diagnosis.json` 的 `flat25.forward_qrms_and_joint_timing` 对每个前进 case 给出全部 12 个腿关节的 request、target、actual 峰值时间和两类误差峰；"
            "其中 `rl_thigh_joint`、`rr_hip_joint`、`fr_calf_joint` 标记为 emphasized。`normal.failure_trajectories` 给出每个失败 case 的命令后首秒、末2秒、最低高度、qRMS、转矩、速度与接触峰值和发生时间。",
            "",
            "`normal.stair_last_2s_jitter` 单独列出每个上下楼梯 case 末2秒的前向速度标准差/峰峰值、高度标准差/峰峰值、跟踪误差、转矩、关节速度、接触和原分析的世界系竖直速度峰。",
            "",
            "V12 student 直接前进中，`rl_thigh_joint` 的 actual–target 误差峰为 0.5780rad@0.34s，actual 相对默认姿态峰为 0.5741rad@0.84s；"
            "`rr_hip_joint` 的 request–target 间隙峰为 0.6012rad@0.34s、actual–target 误差峰为 0.4884rad@0.30s；"
            "`fr_calf_joint` 的 actual–target 误差峰为 0.3780rad@0.06s。settle 前进相应 qRMS 通过，但命令切入附近仍可见 "
            "`fr_calf_joint` actual 峰 0.3072rad@2.44s 和 `rl_thigh_joint` actual–target 误差 0.2056rad@2.04s。",
            "",
            "V12 student 的具体高风险失败包括：上楼梯 L9@1.2m/s 在 6.14s actuator violation，腿速峰 31.5762rad/s@6.14s；"
            "wave L9@0.3m/s 在 0.22s actuator violation，腿速峰 27.4489rad/s；obstacles L6@0.6m/s 在 5.08s illegal contact，"
            "base contact 峰 5131.7N；pit 0.90m@0.6m/s 在 3.42s illegal contact，base contact 峰 5192.4N。"
            "这些是单个固定轨迹的观测值，不外推为随机发生率。",
            "",
            "## 已有图",
            "",
            "- `../pilot300_normal/forward_comparison.png` 与 `../pilot300_normal/startup_joints.png`",
            "- `../pilot300_flat25/forward_comparison.png` 与 `../pilot300_flat25/startup_joints.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="full_episode_v12 receipt directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "trajectory_diagnosis"
    normal_dir = args.root / "pilot300_normal"
    flat_dir = args.root / "pilot300_flat25"
    for directory in (normal_dir, flat_dir):
        for name in ("manifest.json", "analysis.json", "summary.json", "startup_joints.json", "traces.npz"):
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"incomplete input: {path}")
    report = {
        "scope": "fixed cases; not random-trial success probabilities",
        "inputs": {
            directory.name: {
                name: {
                    "bytes": (directory / name).stat().st_size,
                    "sha256": sha256(directory / name),
                }
                for name in ("manifest.json", "analysis.json", "summary.json", "startup_joints.json", "traces.npz")
            }
            for directory in (normal_dir, flat_dir)
        },
        "normal": diagnose_normal(normal_dir),
        "flat25": diagnose_flat25(flat_dir),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "README.md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
