"""Trains the two learnable agents on the canonical benchmark config.

Both are trained on :data:`die.bench.canonical.TRAIN_SEEDS` and never see the
evaluation worlds, so the benchmark measures generalization to unseen food
fields rather than recall of a memorized one.

Each is trained at several seeds and *all* of them are kept. That is not
redundancy: ``docs/JOURNAL.md`` records a seed spread of +/-1.2 to 2.7 on this
benchmark at effect sizes around +2, and two single-seed conclusions that had
to be retracted. A single checkpoint would be an anecdote.

Usage::

    uv run python benchmarks/train_bench.py --out models
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from die.bench.canonical import (
    EVAL_SEEDS,
    EVAL_STEPS,
    FIELD,
    ITERATIONS,
    LEARNING_RATE,
    NUM_ENVS,
    NUM_STEPS,
    TRAIN_REPEATS,
    TRAIN_SEEDS,
    canonical_params,
)
from die.bench.family import dynamic_params
from die.jx.agent import JaxNCAAgent
from die.jx.env import init_state
from die.jx.interface import make_init_bank
from die.jx.rollout import make_evaluate_population
from die.rl.agent import RLAgent
from die.rl.config import HyperParams, TrainStatic
from die.rl.ppo import make_evaluate, make_train
from die.train.evosax_backend import EvosaxBackend

#: Architecture, shared by training and by the benchmark's checkpoint loading.
#: Matches the cross-check's network so returns stay on the same scale.
RL_ARCH = dict(features=16, num_blocks=1, kernel_size=3, hidden=32)

#: Evolution budget. popsize is even because PGPE samples antithetically.
ES_GENERATIONS, ES_POPSIZE = 100, 16
#: NCA action scaling, from the tuned `neural` defaults in die.registry.
NCA_KWARGS = dict(kernel_sizes=(3, 3), scale=0.01, deposit=2.0)


def train_ppo(seed: int, params=None) -> tuple[RLAgent, float]:
    """One PPO run; returns the agent and its held-out return."""
    params = params if params is not None else canonical_params()
    static = TrainStatic(num_envs=NUM_ENVS, num_steps=NUM_STEPS,
                         num_iterations=ITERATIONS, num_epochs=2,
                         num_minibatches=4, bank_size=len(TRAIN_SEEDS),
                         factorization='per_agent', **RL_ARCH)
    hypers = HyperParams(env=params, learning_rate=LEARNING_RATE)

    runner, _ = jax.jit(make_train(static, make_init_bank(TRAIN_SEEDS, params)))(
        jax.random.key(seed), hypers)
    evaluate = jax.jit(make_evaluate(static, make_init_bank(EVAL_SEEDS, params)),
                       static_argnums=(4, 5))
    score = float(evaluate(runner.params, runner.normalizer,
                           jax.random.key(99), hypers, EVAL_STEPS, NUM_ENVS))

    agent = RLAgent(params=runner.params, normalizer=runner.normalizer,
                    action_scale=hypers.action_scale,
                    deposit_scale=hypers.deposit_scale,
                    normalize_obs=static.normalize_obs, **RL_ARCH)
    return agent, score


def train_es(seed: int, params=None) -> tuple[JaxNCAAgent, float]:
    """One evolution run; returns the best agent and its final best fitness.

    The evaluation world rotates through ``TRAIN_SEEDS`` across generations so
    the population is not selected for exploiting one particular food field --
    the batched trainer's ``per_generation`` eval-seed mode, inlined here
    because driving the backend directly avoids constructing an xarray Env
    that this path never steps.
    """
    config = params if params is not None else canonical_params()
    agent = JaxNCAAgent.create(FIELD, seed=seed, **NCA_KWARGS)
    evaluate_population = make_evaluate_population(
        agent.model, config, EVAL_STEPS, agent.action_coefs)
    states = [init_state(int(s), config) for s in TRAIN_SEEDS]

    generation = {'n': 0}

    def evaluate_fitnesses(population) -> np.ndarray:
        state = states[generation['n'] % len(states)]
        return np.asarray(evaluate_population(population, state))

    backend = EvosaxBackend(algorithm='pgpe')
    backend.setup_batched(agent, evaluate_fitnesses, popsize=ES_POPSIZE, seed=seed)

    best = -np.inf
    for _ in range(ES_GENERATIONS):
        stats = backend.run_generation()
        generation['n'] += 1
        best = max(best, stats.best_ever_fitness)
    return backend.best_agent(), float(best)


def score_es_heldout(agent: JaxNCAAgent, params=None) -> float:
    """Mean episode return of an evolved agent on the held-out worlds.

    Reported separately from the training fitness above, which is measured on
    the training worlds and is therefore not comparable with the PPO number.
    """
    config = params if params is not None else canonical_params()
    evaluate_population = make_evaluate_population(
        agent.model, config, EVAL_STEPS, agent.action_coefs)
    single = jax.tree.map(lambda leaf: leaf[None], agent.params)
    returns = [float(evaluate_population(single, init_state(int(s), config))[0])
               for s in EVAL_SEEDS]
    return float(np.mean(returns))


#: The two worlds agents are trained on. Both sets are benchmarked on the whole
#: family, so the table can separate "cannot handle a changing resource" from
#: "was never shown one".
#:
#: A caveat that belongs on the record rather than in a footnote: the two
#: training paths do not see the same physics. Evolution rolls out through
#: ``die.jx.env.step`` -- the physics *core* -- while PPO goes through
#: ``die.jx.interface.step``. Stochastic respawn (like food and comm noise
#: before it) lives in the interface layer to keep the core RNG-free, so the
#: evolved dynamic agents train under regrowth and wave flow but **never under
#: respawn**. The evolved and RL dynamic checkpoints are therefore not trained
#: on identical worlds, and a gap between them partly reflects that.
WORLDS = {'static': lambda: canonical_params(), 'dynamic': dynamic_params}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('models'))
    parser.add_argument('--worlds', nargs='*', default=list(WORLDS),
                        choices=list(WORLDS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary_path = args.out / 'training_summary.json'
    summary: dict[str, dict] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for world in args.worlds:
        params = WORLDS[world]()
        # Static keeps its original bare filenames so existing results, the
        # README examples and the committed checkpoints all still resolve.
        suffix = '' if world == 'static' else f'_{world}'

        for seed in TRAIN_REPEATS:
            started = time.time()
            agent, score = train_ppo(seed, params)
            path = args.out / f'rl_ppo{suffix}_seed{seed}.msgpack'
            agent.save(path)
            summary[path.name] = {'kind': 'ppo', 'world': world, 'seed': seed,
                                  'heldout_return': score,
                                  'seconds': round(time.time() - started, 1)}
            print(f'[ppo/{world}] seed={seed} heldout={score:.3f} -> {path}',
                  flush=True)

        for seed in TRAIN_REPEATS:
            started = time.time()
            agent, fitness = train_es(seed, params)
            heldout = score_es_heldout(agent, params)
            path = args.out / f'jx_nca{suffix}_seed{seed}.msgpack'
            agent.save(path)
            summary[path.name] = {'kind': 'es', 'world': world, 'seed': seed,
                                  'train_fitness': fitness,
                                  'heldout_return': heldout,
                                  'seconds': round(time.time() - started, 1)}
            print(f'[es /{world}] seed={seed} train={fitness:.3f} '
                  f'heldout={heldout:.3f} -> {path}', flush=True)

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
