import json
from pathlib import Path
from statistics import mean

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root / 'runs/mixed9_4096_60000_climbfirst_20260919'
rows = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]
out = {'updates': [rows[0]['completed_updates'], rows[-1]['completed_updates']]}
out['terrain'] = {k: {'count': v['count'], 'first100': mean(r['terrain_by_type'][k]['mean'] for r in rows[:100]), 'last100': mean(r['terrain_by_type'][k]['mean'] for r in rows[-100:]), 'latest': v} for k, v in rows[-1]['terrain_by_type'].items()}
out['course'] = rows[-1]['pit']['course']
out['windows'] = {}
for label, a, b in [('early', rows[0], rows[100]), ('recent', rows[-101], rows[-1])]:
    result = {}
    for who in ['teacher', 'student']:
        ca, cb = a['pit']['climb_by_learner'][who], b['pit']['climb_by_learner'][who]
        result[who + '_pit'] = {k: cb[k]-ca[k] for k in cb}
        ta, tb = a['pit']['by_learner'][who]['totals'], b['pit']['by_learner'][who]['totals']
        result[who + '_pit_causes'] = {k: tb[k]-ta[k] for k in tb}
        for terrain in ['stairs_up', 'stairs_down']:
            sa = a['stair_speed']['episode_statistics'][who][terrain]
            sb = b['stair_speed']['episode_statistics'][who][terrain]
            n = sum(y['episodes']-x['episodes'] for x,y in zip(sa,sb))
            result[who+'_'+terrain] = {'n':n, 'pass': sum(y['radius_passes']-x['radius_passes'] for x,y in zip(sa,sb))/n, 'fail':sum(y['physical_failures']-x['physical_failures'] for x,y in zip(sa,sb))/n, 'vx_mae':sum(y['vx_mae_mps']*y['episodes']-x['vx_mae_mps']*x['episodes'] for x,y in zip(sa,sb))/n}
    out['windows'][label] = result
out['performance'] = {k: mean(r[k] for r in rows[-100:]) for k in ['seconds','collect_seconds','learn_seconds','value','surrogate','entropy','mean_latent_loss','mean_load_balance_loss']}
joints = [j for r in rows[-100:] for j in r['actuator_guard']['joints'].values()]
out['guard_last100'] = {k:sum(j[k] for j in joints) for k in ['overspeed_samples','torque_violation_samples','sustained_overspeed_samples']}
out['guard_last100']['peak_speed'] = max(j['peak_speed'] for j in joints)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run), size_guidance={'scalars':0}).Reload()
    tags = ea.Tags()['scalars']
    out['tb'] = {tag: [(v.step, v.value) for v in ea.Scalars(tag)[-3:]] for tag in tags if any(w in tag.lower() for w in ['learning_rate','std','latent','stability','stationary','termination','reward'])}
except Exception as exc:
    out['tb_error'] = repr(exc)
(run/'audit_progress_1838.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out))
