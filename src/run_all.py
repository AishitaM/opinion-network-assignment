"""Run the whole study end to end: `python src/run_all.py`.

Each stage writes its own artefacts to outputs/, so stages can also be run
individually. The pipeline is deterministic - every random draw is seeded from
config.RANDOM_SEED, so a rerun reproduces every number in the report.
"""
from __future__ import annotations

import time

import analyze
import build_network
import config as C
import prepare_data
import visualize


STAGES = [
    ("1/4  prepare data   ", prepare_data.main),
    ("2/4  build network  ", build_network.main),
    ("3/4  analyse        ", analyze.main),
    ("4/4  visualise      ", visualize.main),
]


def main() -> None:
    print("=" * 74)
    print(f"Opinion Network Formation - pipeline (seed {C.RANDOM_SEED})")
    print("=" * 74)
    t0 = time.time()
    for label, fn in STAGES:
        print(f"\n--- {label} " + "-" * 46)
        t = time.time()
        fn()
        print(f"    done in {time.time() - t:.1f}s")
    print("\n" + "=" * 74)
    print(f"Finished in {time.time() - t0:.1f}s")
    print(f"  figures : {C.FIGS}")
    print(f"  tables  : {C.TABLES}")
    print(f"  network : {C.NETWORK}")
    print("=" * 74)


if __name__ == "__main__":
    main()
