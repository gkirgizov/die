"""The scoring loop: one policy, one scenario, one number.

Every agent is scored through this single path. That is the point of the module
-- the four private return helpers scattered across ``tests/behavior`` each made
their own assumptions about episode length, masking and averaging, and a
comparison built on them would be measuring the harness rather than the agents.

What is measured
----------------
**Total unshaped energy gain, summed over alive agents over a fixed horizon.**
This is the quantity every path in the repo already agrees on: the xarray env
returns it from ``Env.step``, evolution uses it as fitness, and the RL code
carries it as ``raw_return`` precisely so it stays comparable with evolution.
Reward *shaping* is deliberately excluded -- it belongs to the learning layer,
and scoring a scripted agent under someone else's shaping would be meaningless.

Fixed horizon, no auto-reset
----------------------------
Rollouts use :func:`die.jx.interface.step_env` rather than the auto-resetting
``step``. A benchmark wants each world scored on its own merits for exactly
``EVAL_STEPS``; auto-reset would silently mix a fresh world's early, food-rich
steps into a long-running world's score. The canonical config never terminates
mid-episode anyway (``agents_die=False``, ``max_steps=None``), so every episode
here is the same length by construction and returns are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from die.jx.env import init_state
from die.jx.interface import Obs, get_obs, step_env
from die.jx.state import AGENT_ALIVE, MEDIUM_FOOD, EnvParams, EnvState


class Policy(Protocol):
    """A benchmarkable policy: explicit state, functional, jittable.

    Scripted agents in the xarray codebase carry hidden mutable state between
    steps (``GradientAgent._prev_grad`` and friends). Lifting that into an
    explicit carry is what lets a policy run inside ``lax.scan`` and be
    ``vmap``-ed across worlds.
    """

    def init(self, key: jax.Array, params: EnvParams): ...

    def act(self, key: jax.Array, state, obs: Obs, params: EnvParams
            ) -> tuple[jax.Array, object]:
        """Returns ``[3, max_agents]`` action in world units, and the next state."""


@dataclass(frozen=True)
class Score:
    """Per-scenario result, aggregated over world seeds.

    ``return_mean`` is the headline. ``return_sem`` is the standard error over
    seeds, and is not decoration: docs/JOURNAL.md records two conclusions on
    this benchmark that were retracted because a seed spread was mistaken for
    an effect.
    """

    agent: str
    scenario: str
    axis: str
    value: float
    return_mean: float
    return_std: float
    return_sem: float
    per_seed: tuple[float, ...]
    alive_frac: float
    food_consumed: float


def rollout(policy: Policy,
            params: EnvParams,
            state: EnvState,
            key: jax.Array,
            num_steps: int) -> dict[str, jax.Array]:
    """Scores one policy on one world. Returns raw (unaveraged) metrics."""
    food_before = state.medium[MEDIUM_FOOD].sum()

    def scan_step(carry, _):
        env_state, policy_state, step_key = carry
        act_key, noise_key, next_key = jax.random.split(step_key, 3)
        action, policy_state = policy.act(
            act_key, policy_state, get_obs(env_state, params), params)
        timestep, env_state = step_env(noise_key, env_state, action, params)
        # Mask by the pre-step alive flags: dead slots are parked at (0, 0) and
        # their reward entries are not transitions that happened.
        gained = (timestep.reward * timestep.alive).sum()
        return (env_state, policy_state, next_key), (gained, timestep.alive.sum())

    init_key, scan_key = jax.random.split(key)
    policy_state = policy.init(init_key, params)
    (final_state, _, _), (gained, alive) = lax.scan(
        scan_step, (state, policy_state, scan_key), length=num_steps)

    return {
        'total_return': gained.sum(),
        'alive_frac': alive.mean() / jnp.maximum(state.agents[AGENT_ALIVE].sum(), 1.),
        'food_consumed': food_before - final_state.medium[MEDIUM_FOOD].sum(),
    }


def score_scenario(policy: Policy,
                   params: EnvParams,
                   seeds,
                   num_steps: int,
                   key: jax.Array) -> dict[str, np.ndarray]:
    """Runs one policy across a scenario's world seeds.

    Worlds are built on the host (perlin init is not jittable), stacked, and
    rolled out under a single ``vmap`` so a scenario costs one dispatch.
    """
    states = [init_state(int(seed), params) for seed in seeds]
    bank = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    keys = jax.random.split(key, len(states))

    # policy, params and num_steps are captured rather than passed: a policy is
    # a plain Python object with no pytree registration, so vmap could not
    # flatten it as an argument, and num_steps sets a scan trip count and so has
    # to stay static regardless.
    def one_world(state: EnvState, world_key: jax.Array):
        return rollout(policy, params, state, world_key, num_steps)

    metrics = jax.vmap(one_world)(bank, keys)
    return {name: np.asarray(value) for name, value in metrics.items()}
