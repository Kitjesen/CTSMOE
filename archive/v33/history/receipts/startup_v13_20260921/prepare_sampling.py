from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'receipts/full_episode_v12_20260921/next_candidate/pure_start_sampler.py'
target = ROOT/'startup_sampling.py'
assert not target.exists()
text = source.read_text(encoding='utf-8')
text = text.replace('Pure tensor sampler for the prepared flat normal-reset candidate.',
                    'Explicit flat startup coverage for the V13 full-episode experiment.')
target.write_text(text, encoding='utf-8')
