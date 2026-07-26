"""Renders benchmark figures from a results CSV.

    uv run python benchmarks/plot_bench.py --results results

Reads the CSV rather than re-running the sweep, so figures can be reworked
without paying for the rollouts again -- and so the numbers in a figure are
provably the numbers that were measured.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from die.bench.plots import plot_axis_curves, plot_heatmap


def load_rows(path: Path) -> list[dict]:
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['return'] = float(row['return'])
        row['value'] = float(row['value'])
        row['alive_frac'] = float(row['alive_frac'])
        row['food_consumed'] = float(row['food_consumed'])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()

    rows = load_rows(args.results / 'benchmark.csv')
    for path in (plot_heatmap(rows, args.results / 'heatmap.png'),
                 plot_axis_curves(rows, args.results / 'axis_curves.png')):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
