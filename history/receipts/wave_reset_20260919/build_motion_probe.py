"""Reuse the previous matched 72-case evaluator; change only wave placement."""
from pathlib import Path

folder = Path(__file__).resolve().parent
source = (folder.parent/'wave_diagnosis_20260919/probe_wave.py').read_text()
source = source.replace('str(args.root / "code_peak_20260918")', 'str(args.root / "code_wave_reset_20260919")')
before = '    import guard_actions\n'
assert source.count(before) == 1
source = source.replace(before, '    from wave_reset import configure_wave_reset\n'
                        '    configure_wave_reset(cfg, args.wave_reset_profile)\n' + before)
source = source.replace('settle_seconds=args.settle_seconds)',
                        'settle_seconds=args.settle_seconds, wave_reset_profile=args.wave_reset_profile)')
source = source.replace('    parser.add_argument("--reset-z",',
                        '    parser.add_argument("--wave-reset-profile", choices=("original", "wave_surface_v1"), default="wave_surface_v1")\n'
                        '    parser.add_argument("--reset-z",')
(folder/'probe_motion.py').write_text(source)
