import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'code'), str(Path(__file__).resolve().parents[1]/'orchestration')]
import torch
from migrate_routes import migrate
from terrain_allocation import NO_PIT, NO_PIT_PROFILE
from training_profiles import allocate_counts


def test_migration_retains_old_state_and_adds_named_routes():
    names = ['wave']*3+['slope_up']+['slope_down']+['rough_slope']+['stairs_up']*5+['stairs_down']*2+['obstacles']*4+['flat']*3+['pit']*20
    old_weights = {k:v/.8 for k,v in NO_PIT.items() if not k.startswith('continuous_')}
    columns = torch.zeros(8192,dtype=torch.long)
    for learner in (0,1):
        ids = [i for i in range(8192) if int(i%4==0)==learner]
        counts = allocate_counts(len(ids), old_weights)
        cursor=0
        for name,count in counts.items():
            if count:
                columns[ids[cursor:cursor+count]]=names.index(name)
                cursor+=count
    saved=dict(terrain_levels=torch.arange(8192)%10,terrain_types=columns,terrain_column_names=names,
               model_state_dict={'w':torch.ones(3)},optimizer_state_dict={'step':10},optimizer_stu_enc_state_dict={'step':9},
               terrain_allocation=dict(pending={},stats=torch.ones(9,22,6,3,3),course={'bad_streak':{},'weights':old_weights,'profile':'omni_eight_v1'}))
    out,evidence=migrate(saved)
    assert 1630 <= evidence['moved'] <=1645
    assert out['terrain_column_names'][:20] == names[:20]
    assert out['terrain_allocation']['course']['profile']==NO_PIT_PROFILE
    assert out['terrain_allocation']['stats'][:9].eq(1).all()
    assert out['terrain_allocation']['stats'][9:].eq(0).all()
    assert (out['terrain_levels'][out['terrain_types']>=20]==0).all()

def test_continuous_fixed_allocation_bounds_and_evaluation():
    from terrain_allocation import AllocationCourse, BASE
    course=AllocationCourse(profile=NO_PIT_PROFILE,weights=dict(NO_PIT))
    result=course.evaluate({name:{} for name in BASE},500)
    assert result['move'] is None
    assert result['weights']==NO_PIT
