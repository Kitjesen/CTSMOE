"""Summarize traced episodes separately from partial/censored probe episodes."""
import json
from pathlib import Path
import sys

import numpy as np


def summarize(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    result = dict(steps=d['steps'], optimizer_updates=d['optimizer_updates'],
                  hashes_unchanged=d['hashes_before'] == d['hashes_after'],
                  hold_mismatches=d['hold_mismatches'], groups={})
    delta = np.asarray(d['totals_delta'])
    # Counter shape is learner,level,speed_bin,statistic in the production code.
    if delta.shape != (2, 22, 4, 12):
        raise ValueError(f'Unexpected counter shape {delta.shape}')
    for learner, label in [(0, 'teacher'), (1, 'student')]:
        for level in [None, 2]:
            cs = [c for c in d['episodes'] if c['learner'] == learner
                  and (level is None or c['level'] == level)]
            tag = label + ('_all_heights' if level is None else '_25cm')
            if not cs:
                result['groups'][tag] = {'episodes': 0}
                continue
            counts = np.asarray([c['conditions_steps'] for c in cs])
            longest = np.asarray([c['longest_steps'] for c in cs])
            bad = np.asarray([c['post_stop_bad_steps'] for c in cs])
            resets = np.asarray([c['hold_reset_reasons'] for c in cs])
            observed_stop = int(counts[:, 7].sum() - (counts[:, 7] > 0).sum())
            total = delta[learner].sum((0, 1)) if level is None else delta[learner, level].sum(0)
            checks = [int(total[0]) == len(cs), int(total[1]) == sum(c['success'] for c in cs),
                      int(total[2]) == sum(c['physical'] for c in cs),
                      int(total[7]) == sum(c['outside'] for c in cs)]
            if not all(checks):
                raise AssertionError(f'Production counters differ from recorded episodes: {tag}')
            result['groups'][tag] = dict(episodes=len(cs), counters_match=True,
                success=sum(c['success'] for c in cs), physical=sum(c['physical'] for c in cs),
                outside=sum(c['outside'] for c in cs),
                ever_by_condition=dict(zip(d['conditions'], (counts > 0).sum(0).tolist())),
                max_consecutive_s=dict(zip(d['conditions'], (longest.max(0)*.02).tolist())),
                max_hold_s=max(c['max_hold_s'] for c in cs),
                median_max_hold_s=float(np.median([c['max_hold_s'] for c in cs])),
                post_stop_observed_steps=observed_stop,
                post_stop_bad_steps=dict(zip(d['post_stop_bad_conditions'], bad.sum(0).tolist())),
                post_stop_bad_fraction=dict(zip(d['post_stop_bad_conditions'],
                    (bad.sum(0)/max(1, observed_stop)).tolist())),
                hold_reset_reason_counts=dict(zip(d['post_stop_bad_conditions'], resets.sum(0).tolist())),
                command_mismatch_steps=sum(c['command_mismatch_steps'] for c in cs))
    return result


if __name__ == '__main__':
    report = summarize(sys.argv[1])
    print(json.dumps(report, indent=2))
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding='utf-8')
