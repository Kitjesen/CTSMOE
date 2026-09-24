"""Paired GPU timings; twelve buffers match the four Thunder actuator groups."""
import argparse
import json
import statistics
import time

import torch

from fast_delay import FastDelayBuffer
from test_fast_delay import reference_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()
    cls = reference_type()
    widths = [4, 4, 4, 4]
    generators = {}
    inputs = [torch.ones(args.envs, width, device="cuda") for width in widths for _ in range(3)]
    ids = torch.arange(0, args.envs, 20, device="cuda")
    for mode in ("reference", "fast"):
        buffers = []
        for _ in inputs:
            b = cls(4, args.envs, "cuda:0")
            if mode == "fast":
                b = FastDelayBuffer(b)
            b.set_time_lag(torch.arange(args.envs, device="cuda", dtype=torch.int) % 5)
            for _ in range(6):
                b.compute(inputs[len(buffers)])
            buffers.append(b)
        generators[mode] = buffers
    results = {mode: [] for mode in generators}
    for repetition in range(6):
        order = ("reference", "fast") if repetition % 2 == 0 else ("fast", "reference")
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(args.steps):
                for b, data in zip(generators[mode], inputs):
                    if step % 16 == 0:
                        b.reset(ids)
                    b.compute(data)
            torch.cuda.synchronize()
            results[mode].append(time.perf_counter() - start)
    medians = {k: statistics.median(v) for k, v in results.items()}
    print(json.dumps(dict(envs=args.envs, steps=args.steps, seconds=results, medians=medians,
                          speedup=medians["reference"] / medians["fast"]), indent=2))


if __name__ == "__main__":
    main()
