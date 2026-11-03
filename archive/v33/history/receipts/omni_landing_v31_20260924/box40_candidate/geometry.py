"""CPU candidate only: signed boxes with at most 40 cm adjacent height change.

Not installed in the frozen V31 run. Heights remain signed; tall bumps and deep
depressions must be separated by intermediate ground if directly incompatible.
"""
import numpy as np


def height_field(difficulty: float, seed: int) -> np.ndarray:
    if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError('difficulty must lie in [0, 1]')
    rng = np.random.default_rng(seed)
    # Existing 9 m tile minus two 0.5 m borders, 10 cm horizontal / 5 mm vertical grid.
    pixels = 80
    field = np.zeros((pixels, pixels), dtype=np.int16)
    height = int(round((.05 + .35 * difficulty) / .005))
    adjacent_limit = 80  # 40 cm / 5 mm.
    for _ in range(20):
        width, length = rng.choice([10, 14, 18], size=2)
        x, y = rng.choice(np.arange(0, pixels, 4), size=2)
        x, y = min(int(x), pixels-width), min(int(y), pixels-length)
        a, b = max(0, x-1), min(pixels, x+width+1)
        c, d = max(0, y-1), min(pixels, y+length+1)
        halo = field[a:b, c:d]
        mask = np.ones(halo.shape, dtype=bool)
        mask[x-a:x-a+width, y-c:y-c+length] = False
        neighbor = halo[mask].astype(np.int32)
        low = max(-height, int(neighbor.max())-adjacent_limit)
        high = min(height, int(neighbor.min())+adjacent_limit)
        if low > high:
            raise ValueError('No admissible box height at this placement')
        proposed = int(rng.choice([-height, -height//2, height//2, height]))
        # Restrict this one rectangular block to its actual neighboring surfaces.
        field[x:x+width, y:y+length] = np.clip(proposed, low, high)
    field[25:55, 25:55] = 0  # Existing 3 m central spawn platform.
    return np.pad(field, 5)  # Existing 0.5 m zero-height tile border.


def max_adjacent_step_m(field: np.ndarray) -> float:
    values = field.astype(np.int32)
    differences = [np.diff(values, axis=0), np.diff(values, axis=1),
                   values[1:,1:]-values[:-1,:-1], values[1:,:-1]-values[:-1,1:]]
    return max(float(np.abs(x).max()) for x in differences) * .005
