"""Matched stair/pit regression before and after a bounded repair candidate."""
from pathlib import Path

p=Path(__file__).parent
s=(p.parents[1]/'recordings/stairs_strong_20260920/record.py').read_text(encoding='utf-8')
s=s.replace("for key in ('root','output'):","for key in ('root','output','selection'):")
s=s.replace("receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())", "receipt=json.loads((a.root/'stairs_video_snapshot.json').read_text())\nselection=json.loads(a.selection.read_text())\nreceipt['checkpoint']=str(a.root/selection[1]['path'])")
s=s.replace("before_path=a.root/'runs/mixed9_4096_60000_shared_posture_20260920/model_41750.pt'", "before_path=a.root/selection[0]['path']")
s=s.replace("cases += [dict(policy='current',terrain='pit',level=21,speed=speed,settle_s=0.) for speed in (.3,.6,.9)]", "cases += [dict(policy=policy,terrain='pit',level=21,speed=speed,settle_s=0.) for policy in ('before','current') for speed in (.3,.6,.9)]")
s=s.replace("print('DONE',json.dumps(cases),flush=True)","print('DONE',len(cases),flush=True)")
(p/'regression.py').write_text(s,encoding='utf-8')
