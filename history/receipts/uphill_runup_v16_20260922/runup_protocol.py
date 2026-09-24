"""Paired uphill diagnostics; every comparison uses the same physical slope exit."""
from itertools import product


def build_cases(selection, suite):
    if suite != "terrain83":
        raise ValueError("Run-up diagnostic uses slope terrain only")
    cases = []
    for policy, item in enumerate(selection):
        for level, speed, gap, lateral in product((6, 7, 9), (.3, .6, 1., 1.2, 1.5), (.1, 1.5), (-.1, 0., .1)):
            cases.append(dict(id=len(cases), policy=policy, teacher=item["teacher"],
                              terrain="slope_up", level=level, mode="direct", direction="forward",
                              velocity=[speed, 0., 0.], suite="uphill_runup", gap=gap, lateral=lateral))
    return cases


def summarize(manifest):
    """Fixed-case counts, not random success probabilities or deployment acceptance."""
    cases = manifest["cases"]
    groups = []
    for teacher, level, speed, gap in product((False, True), (6, 7, 9), (.3, .6, 1., 1.2, 1.5), (.1, 1.5)):
        rows = [r for r in cases if (r["teacher"], r["level"], r["velocity"][0], r["gap"]) == (teacher, level, speed, gap)]
        if len(rows) != 3:
            raise ValueError("Missing matched lateral-offset cases")
        groups.append(dict(teacher=teacher, level=level, speed=speed, gap=gap, total=3,
                           passed=sum(r["passed_4m"] for r in rows),
                           physical=sum(r["physical"] for r in rows),
                           entry_speeds=[r.get("entry_speed_mps") for r in rows],
                           exit_x_m=[r["dx_m"] for r in rows]))
    return dict(groups=groups, fixed_cases_not_random_success_rate=True,
                criterion="base reaches x=4m relative to tile center without physical failure; not whole-body crest completion")
