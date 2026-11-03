"""Review-only asymmetric terrain meshes; no simulator registration."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import trimesh
from routes import triangulate_polygon


def strip(points, low_y, high_y):
    side = np.asarray(points + [(points[-1][0], -.6), (0., -.6)])
    n = len(side)
    vertices = [(x, y, z) for y in (low_y, high_y) for x, z in side]
    triangles = triangulate_polygon(side)
    faces = list(triangles) + [tuple(n+i for i in reversed(t)) for t in triangles]
    for i in range(n):
        j = (i+1) % n
        faces.extend(((i, j, n+j), (i, n+j, n+i)))
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    m.fix_normals()
    return m


def cross_slope():
    xs, ys = np.linspace(0, 7, 141), np.linspace(-1.5, 1.5, 61)
    vertices = []
    for x in xs:
        for y in ys:
            ramp = np.clip((x-1.2)/.8, 0, 1) * np.clip((6.5-x)/.8, 0, 1)
            z = y*np.tan(np.deg2rad(12))*ramp
            # Compact raised-cosine bumps preserve a single closed surface.
            for bx, by in [(2.5,.25),(3.7,-.25),(4.9,.25)]:
                dx, dy = abs(x-bx)/.3, abs(y-by)/.22
                if dx < 1 and dy < 1:
                    z += .10*(1+np.cos(np.pi*dx))*(1+np.cos(np.pi*dy))/4
            vertices.append((x,y,z))
    faces = []
    ny = len(ys)
    for i in range(len(xs)-1):
        for j in range(ny-1):
            a = i*ny+j
            faces.extend(((a,a+ny,a+ny+1),(a,a+ny+1,a+1)))
    boundary = ([i*ny for i in range(len(xs))]
                + [(len(xs)-1)*ny+j for j in range(1,ny)]
                + [i*ny+ny-1 for i in range(len(xs)-2,-1,-1)]
                + list(range(ny-2,0,-1)))
    lower = []
    for idx in boundary:
        lower.append(len(vertices))
        x,y,_ = vertices[idx]
        vertices.append((x,y,-.6))
    center = len(vertices)
    vertices.append((3.5,0,-.6))
    for k,a in enumerate(boundary):
        j = (k+1) % len(boundary)
        b = boundary[j]
        faces.extend(((a,b,lower[j]),(a,lower[j],lower[k]),(center,lower[k],lower[j])))
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    m.fix_normals()
    return m


def make(kind):
    flat = [(0,0),(7,0)]
    high = [(0,0),(1.2,0),(2,.2),(5.5,.2),(6.3,0),(7,0)]
    ramp = [(0,0),(1.5,0),(4.5,.6),(5.5,.6),(6.5,0),(7,0)]
    stairs = [(0,0),(2,0),(2,.2),(2.8,.2),(2.8,.4),(3.6,.4),(3.6,.6),(5.5,.6),(6.5,0),(7,0)]
    if kind == 'cross_slope_bumps':
        return cross_slope(), '12°横坡，三个10厘米圆滑凸块；左右交替布置。', 12, .10
    if kind == 'left_high':
        a,b,desc = high,flat,'左侧高20厘米、右侧平地；入口和出口通过缓坡连接。'
    elif kind == 'right_high':
        a,b,desc = flat,high,'右侧高20厘米、左侧平地；与左高版本镜像。'
    elif kind == 'split_ramp_stairs':
        a,b,desc = ramp,stairs,'左侧连续坡，右侧三级20厘米台阶；最终同高60厘米。'
    elif kind == 'offset_ramp_stairs':
        a = [(0,0),(2.7,0),(3.5,.3),(4.3,.3),(5.1,0),(7,0)]
        b = [(0,0),(1.8,0),(1.8,.2),(2.6,.2),(2.6,.4),(3.6,.4),(3.6,.2),(4.4,.2),(4.4,0),(7,0)]
        desc = '坡段与台阶前后错开，产生前后轮不同接触组合；具体单轮状态需按实际轴距验证。'
    else:
        raise ValueError(kind)
    # Adjacent closed strips have coincident internal walls. Export is a review
    # assembly, not a collision-union approved for training.
    m = trimesh.util.concatenate((strip(a,0,1.5), strip(b,-1.5,0)))
    return m,desc,0,.2


def main():
    out = Path(__file__).parent/'assets'
    page = out/'index.html'
    text = page.read_text(encoding='utf8')
    start = text.index('const scenes=')+len('const scenes=')
    end = text.index(';\nconst canvas',start)
    scenes = json.loads(text[start:end])[:4]
    names = [('left_high','左高右低 · 20 cm'),('right_high','右高左低 · 20 cm'),
             ('cross_slope_bumps','横坡12°＋交替小凸块'),('split_ramp_stairs','左轮坡面＋右轮台阶'),
             ('offset_ramp_stairs','错位坡面＋台阶 · 单轮接触组合')]
    for index,(kind,name) in enumerate(names,5):
        mesh,desc,angle,h = make(kind)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        assert mesh.area_faces.min() > 1e-10
        file = f'{index}_{kind}.obj'
        mesh.export(out/file)
        data = dict(name=name,description=desc,units='m',up_axis='Z',forward_axis='X',
                    config=dict(width=3,kind='asymmetric',count=1,slope_deg=angle),
                    length_m=7,max_step_m=h,training_integrated=False,
                    mesh_contract='closed visual assembly; seam collision validation pending',
                    sha256=hashlib.sha256((out/file).read_bytes()).hexdigest())
        (out/f'{index}_{kind}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        scenes.append(dict(name=name,vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist(),meta=data,file=file))
    text = text[:start]+json.dumps(scenes,ensure_ascii=False)+text[end:]
    text = text.replace("c.kind==='ramp_stairs_ramp'?c.slope_deg+'°':'—'", "c.slope_deg?c.slope_deg+'°':'—'")
    text = text.replace('最大实际阶差', '阶差／凸起标称高度')
    if '${m.description' not in text:
        text = text.replace('重复数量 <b>${c.count}</b>', '重复数量 <b>${c.count}</b><p>${m.description||""}</p>')
    if '左右分界在Y=0' not in text:
        text = text.replace('当前未验证通行能力。', '当前未验证通行能力。不对称地形以X前进、Y正方向为左侧；左右分界在Y=0。')
    text = text.replace('continuous_maps_v2.zip','continuous_maps_v3.zip')
    text = text.replace("c.slope_deg?c.slope_deg+'°':'—'", "(c.kind==='ramp_stairs_ramp'||c.kind==='asymmetric')&&c.slope_deg?c.slope_deg+'°':'—'")
    text = text.replace('function render(){const rect=', "function render(){const showWire=document.querySelector('#wire').checked;const rect=")
    text = text.replace("let edge=document.querySelector('#wire').checked&&", 'let edge=showWire&&')
    page.write_text(text,encoding='utf8')
    with zipfile.ZipFile(out/'continuous_maps_v3.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in ('.obj','.json','.html','.png'):
                z.write(f,f.name)
    print('Exported 5 asymmetric meshes; 9 viewer scenes total')


if __name__ == '__main__':
    main()
