"""Scores every agent across the whole environment family.

    uv run python benchmarks/bench_agents.py --out results

Writes a tidy CSV (one row per agent x scenario x seed) plus an aggregated
JSON. Tidy because every downstream question -- per-axis curves, seed spreads,
agent rankings -- is a groupby on that shape, and because storing only the
aggregate would throw away the per-seed variation that this benchmark's own
history says is the easiest thing to mistake for an effect.

Every trained agent appears once per training seed rather than being averaged
into a single row. Averaging them here would hide exactly the spread the
journal keeps warning about.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import numpy as np

from die.bench.canonical import EVAL_STEPS
from die.bench.family import FAMILY_SEEDS, build_family
from die.bench.policies import load_learned
from die.bench.runner import score_scenario
from die.jx.scripted import (
    BrownianPolicy,
    GradientPolicy,
    NullPolicy,
    PhysarumPolicy,
)

#: Scripted baselines, at the tuned defaults from die.registry's make_agent.
#: `null` is the bar that matters: not "better than random" but "better than
#: inaction", which on this env is a genuinely strong baseline -- agents burn
#: energy to move, so any policy that moves without finding food loses to it.
SCRIPTED = {
    'null': NullPolicy(),
    'brownian': BrownianPolicy(),
    'gradient': GradientPolicy(),
    'physarum': PhysarumPolicy(),
}


def discover_learned(models: Path) -> dict:
    """Every checkpoint in the models directory, keyed by filename stem."""
    found = {}
    for path in sorted(models.glob('*.msgpack')):
        found[path.stem] = load_learned(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--steps', type=int, default=EVAL_STEPS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policies = dict(SCRIPTED)
    policies.update(discover_learned(args.models))
    family = build_family()
    print(f'{len(policies)} agents x {len(family)} scenarios '
          f'x {len(FAMILY_SEEDS)} seeds', flush=True)

    rows, started = [], time.time()
    for agent_name, policy in policies.items():
        for scenario in family:
            metrics = score_scenario(policy, scenario.params(), FAMILY_SEEDS,
                                     args.steps, jax.random.key(0))
            for index, seed in enumerate(FAMILY_SEEDS):
                rows.append({
                    'agent': agent_name,
                    'scenario': scenario.name,
                    'axis': scenario.axis,
                    'value': scenario.value,
                    'seed': seed,
                    'return': float(metrics['total_return'][index]),
                    'alive_frac': float(metrics['alive_frac'][index]),
                    'food_consumed': float(metrics['food_consumed'][index]),
                })
        done = np.mean([r['return'] for r in rows if r['agent'] == agent_name])
        print(f'  {agent_name:<16} family mean {done:7.3f}', flush=True)

    csv_path = args.out / 'benchmark.csv'
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate with the standard error over seeds alongside the mean: the
    # spread is the part a reader needs in order not to over-read a gap.
    aggregate: dict[str, dict] = {}
    for agent_name in policies:
        aggregate[agent_name] = {}
        for scenario in family:
            values = np.array([r['return'] for r in rows
                               if r['agent'] == agent_name
                               and r['scenario'] == scenario.name])
            aggregate[agent_name][scenario.name] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)),
                'sem': float(values.std(ddof=1) / np.sqrt(len(values))),
            }
    (args.out / 'benchmark.json').write_text(json.dumps(aggregate, indent=2))

    print(f'\n{len(rows)} rows -> {csv_path}')
    print(f'elapsed {time.time() - started:.1f}s')


if __name__ == '__main__':
    main()
