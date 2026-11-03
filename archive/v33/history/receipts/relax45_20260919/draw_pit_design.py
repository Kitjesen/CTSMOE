"""Draw the actual box layout; this is a design diagram, not a simulation screenshot."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pit_geometry import pit_boxes, ORIGIN, EDGE_X  # noqa: E402

font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
fig, (top, side) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.15]})
fig.patch.set_facecolor('#f8fafc')
colors = {'floor':'#edf2f7', 'platform':'#b7d9ec', 'back':'#64748b', 'left':'#64748b', 'right':'#64748b'}
top.add_patch(Rectangle((0,0),9,9,facecolor='#edf2f7',edgecolor='#94a3b8'))
for name, (lo, hi) in pit_boxes(.25).items():
    if name == 'floor':
        continue
    top.add_patch(Rectangle((lo[0]+.5,lo[1]+1.5),hi[0]-lo[0],hi[1]-lo[1],facecolor=colors[name]))
top.add_patch(Rectangle((2.7,3.5),4.8,2,fill=False,edgecolor='#d97706',linestyle='--',linewidth=1.5))
top.plot(ORIGIN[0]+.5,ORIGIN[1]+1.5,'o',color='#0f766e',markersize=9)
top.annotate('',xy=(6.5,4.5),xytext=(3.65,4.5),arrowprops={'arrowstyle':'->','lw':2.5,'color':'#0f766e'})
top.text(3.5,4.05,'出生点',ha='center',fontsize=11)
top.text(2.7,6.3,'坑内净空约 4.3 × 5.6 m',ha='center',fontsize=11)
top.text(6.7,6.3,'高台顶面\n3.5 × 6 m',ha='center',fontsize=12)
top.text(5.1,3.1,'橙框：2 m 宽有效训练区域',ha='center',fontsize=10,color='#b45309')
top.text(5.12,2.1,'垂直台阶边缘',rotation=90,fontsize=10)
top.set(xlim=(-.2,9.2),ylim=(-.2,9.2),xlabel='前进方向 x（m）',ylabel='横向 y（m）',title='俯视：8 × 6 m 坑体嵌入 9 × 9 m 地形格')
top.set_aspect('equal')

side.add_patch(Rectangle((0,-.05),8,.05,facecolor='#64748b'))
side.add_patch(Rectangle((0,0),.2,.25,facecolor='#64748b'))
side.add_patch(Rectangle((EDGE_X,0),8-EDGE_X,.25,facecolor='#b7d9ec',edgecolor='#256f98'))
side.plot(3,.04,'o',color='#0f766e',markersize=9)
side.annotate('',xy=(4.4,.045),xytext=(3.15,.045),arrowprops={'arrowstyle':'->','lw':2,'color':'#0f766e'})
side.annotate('',xy=(4.5,.14),xytext=(3,.14),arrowprops={'arrowstyle':'<->','color':'#475569'})
side.text(3.75,.16,'名义距离 1.5 m',ha='center',fontsize=11)
side.annotate('',xy=(8.15,.25),xytext=(8.15,0),arrowprops={'arrowstyle':'<->','color':'#256f98'})
side.text(6.2,.34,'当前课程前沿：H = 0.25 m',ha='center',fontsize=12)
side.text(6.2,.29,'前方为单级直立面',ha='center',fontsize=11)
side.text(4.3,.53,'高度课程：0.15 → 1.20 m，每档增加 0.05 m',ha='center',fontsize=12)
side.text(4.3,.46,'前进爬上平台 → 全身越过边缘 → 停稳 2 秒',ha='center',fontsize=11)
side.set(xlim=(-.1,8.6),ylim=(-.08,.62),xlabel='坑体局部 x（m）',ylabel='高度（m）',title='侧视：当前25厘米档（竖直方向放大显示）')
for ax in (top,side):
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Thunder 高台／出坑：由当前碰撞几何代码生成的设计示意',fontsize=16)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(Path(__file__).with_name('pit_design.png'),dpi=160,facecolor=fig.get_facecolor())
