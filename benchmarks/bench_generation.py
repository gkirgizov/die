"""Walltime of one evolution generation: evotorch (torch/xarray) vs evosax (JAX).

Not part of the test suite -- it is a sanity check on the speedup claim in the
modernization roadmap (docs/NOTES.md), which motivated the JAX port.

Compile time is reported separately from steady-state generation time: the JAX
path pays a one-off XLA compilation on the first generation, so only the
steady-state number is meaningful for a long run.

Usage:
    uv run python benchmarks/bench_generation.py --field-size 64 --popsize 16
"""

from __future__ import annotations

import argparse
import time


def bench_evosax(field_size: int, popsize: int, episode_steps: int,
                 generations: int, seed: int) -> tuple[float, float]:
    from die.env import Dynamics, Env
    from die.jx.agent import JaxNCAAgent
    from die.jx.factory import make_jx_config
    from die.train.evosax_backend import EvosaxBackend
    from die.train.jax_trainer import JaxEvolutionTrainer
    from die.train.trainer import TrainConfig

    field = (field_size, field_size)
    env = Env(field, Dynamics())
    agent = JaxNCAAgent.create(field, seed=seed, kernel_sizes=(3, 3),
                               scale=0.01, deposit=2.0)
    config = TrainConfig(generations=generations, episode_steps=episode_steps,
                         popsize=popsize, seed=seed, checkpoint_every=0, record_every=0)
    trainer = JaxEvolutionTrainer(env, agent, EvosaxBackend('pgpe'), config,
                                  jx_config=make_jx_config('st-perlin-wide', field_size))
    trainer._setup_backend()

    start = time.perf_counter()
    trainer.backend.run_generation()          # includes XLA compilation
    first = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(generations):
        trainer.backend.run_generation()
    steady = (time.perf_counter() - start) / generations
    return first, steady


def bench_evotorch(field_size: int, popsize: int, episode_steps: int,
                   generations: int, seed: int) -> tuple[float, float]:
    from die.registry import AgentId, make_agent, make_env
    from die.train.evotorch_backend import EvotorchBackend
    from die.train.trainer import EvolutionTrainer, TrainConfig

    env = make_env('st-perlin-wide', field_size)
    agent = make_agent(AgentId.neural, field_size, seed=seed)
    config = TrainConfig(generations=generations, episode_steps=episode_steps,
                         popsize=popsize, seed=seed, checkpoint_every=0, record_every=0)
    trainer = EvolutionTrainer(env, agent, EvotorchBackend('pgpe'), config)
    trainer._setup_backend()

    start = time.perf_counter()
    trainer.backend.run_generation()
    first = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(generations):
        trainer.backend.run_generation()
    steady = (time.perf_counter() - start) / generations
    return first, steady


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field-size', type=int, default=64)
    parser.add_argument('--popsize', type=int, default=16)
    parser.add_argument('--episode-steps', type=int, default=50)
    parser.add_argument('--generations', type=int, default=3,
                        help='Generations timed for the steady-state average.')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    print(f'field={args.field_size}x{args.field_size}, popsize={args.popsize}, '
          f'episode_steps={args.episode_steps}, averaged over {args.generations} generations')

    results = {}
    for name, fn in (('evosax (JAX)', bench_evosax), ('evotorch (torch)', bench_evotorch)):
        try:
            first, steady = fn(args.field_size, args.popsize, args.episode_steps,
                               args.generations, args.seed)
        except ImportError as exc:
            print(f'{name}: skipped ({exc})')
            continue
        results[name] = steady
        print(f'{name}: first generation {first:.2f}s (incl. compile), '
              f'steady state {steady:.2f}s/generation')

    if len(results) == 2:
        speedup = results['evotorch (torch)'] / results['evosax (JAX)']
        print(f'\nSteady-state speedup: {speedup:.1f}x')


if __name__ == '__main__':
    main()
