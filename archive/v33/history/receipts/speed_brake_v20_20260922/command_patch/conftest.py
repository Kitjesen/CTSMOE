"""Use the frozen V19 boundary dependency without modifying its receipt."""
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / 'omni_repair_v19_20260922' / 'recovery01' / 'code'
sys.path.append(str(BASELINE))
