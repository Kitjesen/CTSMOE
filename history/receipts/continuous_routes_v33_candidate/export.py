"""Export review assets and plots from the actual collision geometry."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from routes import Course, generate


def main():
    out = Path(__file__).parent / 'assets'
    out.mkdir(exist_ok=True)
    recipes = [Course('curbs', height=.4), Course('blocks', height=.4, count=4, seed=7),
               Course('ramp_stairs_ramp', height=.25, slope_deg=35),
               Course('ramp_stairs_ramp', height=.3, slope_deg=45)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), constrained_layout=True)
    manifest = []
    for i, (cfg, ax) in enumerate(zip(recipes, axes)):
        mesh, data = generate(cfg)
        name = f'{i+1}_{cfg.kind}'
        file = out / (name + '.obj')
        mesh.export(file)
        data['obj_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        (out / (name + '.json')).write_text(json.dumps(data, indent=2))
        manifest.append(data)
        x, z = zip(*data['profile'])
        ax.fill_between(x, z, -.25, color='#197f85')
        ax.plot(x, z, color='#073b4c', linewidth=2)
        ax.set(title=f'{name} | step {cfg.height*100:.0f} cm | slope {cfg.slope_deg if cfg.kind == "ramp_stairs_ramp" else 0:.0f} deg',
               xlabel='Forward distance (m)', ylabel='Height (m)', ylim=(-.3, 2.))
        ax.grid(alpha=.2)
    fig.suptitle('Candidate collision mesh profiles — not yet integrated into training')
    fig.savefig(out/'profiles.png', dpi=160)
    plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
