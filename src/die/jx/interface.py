"""Functional, RL-shaped interface over the JAX env physics.

Shaped after the gymnax convention, which is what the JAX RL ecosystem
converged on -- ``reset(key, params)`` / ``step(key, state, action, params)``,
with parameters as a pytree kept separate from state. Two of its patterns are
worth copying and are copied here: the public ``step`` owns **auto-reset**, so
episode-boundary handling never leaks into rollout code, and ``params`` stays
separate from ``state`` so a population of configurations is ``vmap``-able.

Three deliberate deviations, each forced by what this environment actually is:

- **Reward and terminals are ``[max_agents]`` arrays, not scalars.** There is one
  policy instantiated across a population of locations, so a world step yields
  hundreds of transitions rather than one. JaxMARL-style per-agent dicts are the
  wrong shape for thousands of homogeneous agents.
- **Termination and truncation are returned separately.** Collapsing them into a
  single ``done``, as gymnax does, loses the distinction GAE needs: a truncated
  episode must bootstrap its value estimate, a terminated one must not.
- **``reset`` draws from a pre-built bank of initial worlds** rather than
  generating one. Initialization runs perlin noise through the host-side
  :class:`~die.data_init.DataInitializer` and cannot be jitted; a stacked bank
  makes the choice a traced index instead. This generalizes the per-seed caches
  already in :attr:`die.env.Env._init_cache` and
  :attr:`die.train.jax_trainer.JaxEvolutionTrainer._state_cache`.

Reward here is the *raw* per-agent energy change. Shaping it -- collective vs
selfish sharing, survival bonuses -- belongs to the learning layer in
``die.rl``, not to the physics: :class:`~die.jx.state.EnvParams` mirrors
:class:`die.env.Dynamics`, and shaping has no twin there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from die.jx.env import init_state, step_with_aux
from die.jx.state import AGENT_ALIVE, AGENT_FOOD, EnvParams, EnvState


@struct.dataclass
class Obs:
    """What the policy perceives.

    :param medium: ``[3, width, height]`` -- the full field. A convolutional
        policy reads this once per world step and each agent gathers at its own
        cell, so the *effective* per-agent observation is the receptive field
        around it. That is the sense in which this stays a local policy.
    :param agent: ``[2, max_agents]`` proprioception -- (agent_food, alive).
        The field alone does not tell an agent its own energy, which is exactly
        the quantity survival depends on.
    """

    medium: jax.Array
    agent: jax.Array


@struct.dataclass
class TimeStep:
    """One environment transition, per agent.

    :param obs: observation *after* the step (post-auto-reset, when it fired).
    :param reward: ``[max_agents]`` raw energy change; unshaped.
    :param terminated: ``[max_agents]`` the agent died this step. No bootstrap.
    :param truncated: scalar; the world hit ``max_steps`` or went extinct.
        Bootstrap normally -- the episode was cut, not completed.
    :param alive: ``[max_agents]`` alive mask *before* the step: the agents that
        actually acted, and so the only ones whose transitions are real. Every
        loss reduction, advantage normalization and GAE accumulation must mask
        by this or dead slots phantom-participate.
    """

    obs: Obs
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    alive: jax.Array


def get_obs(state: EnvState, params: EnvParams) -> Obs:
    """Assembles the observation from world state. No normalization here."""
    del params      # obs shape is fixed by the state itself
    return Obs(medium=state.medium,
               agent=state.agents[jnp.array([AGENT_FOOD, AGENT_ALIVE])])


def make_init_bank(seeds, params: EnvParams) -> EnvState:
    """Stacks K initial worlds into one pytree with a leading bank axis.

    Built on the host, once, because perlin initialization is expensive and not
    jittable. Every seed goes through the same :class:`DataInitializer` pipeline
    the xarray engine uses, so worlds stay comparable seed-for-seed.
    """
    states = [init_state(int(seed), params) for seed in seeds]
    if not states:
        raise ValueError('An init bank needs at least one seed')
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)


def reset(key: jax.Array, bank: EnvState, params: EnvParams) -> tuple[Obs, EnvState]:
    """Starts a fresh episode on a world drawn uniformly from the bank."""
    size = jax.tree.leaves(bank)[0].shape[0]
    index = jax.random.randint(key, (), 0, size)
    state = jax.tree.map(lambda leaf: leaf[index], bank)
    return get_obs(state, params), state


def step_env(state: EnvState,
             action: jax.Array,
             params: EnvParams) -> tuple[TimeStep, EnvState]:
    """One step of raw physics, without auto-reset.

    ``action`` is ``[3, max_agents]`` -- (dx, dy, deposit1), already scaled into
    world units by the caller.
    """
    next_state, _, aux = step_with_aux(state, action, params)

    extinct = next_state.agents[AGENT_ALIVE].sum() == 0
    if params.max_steps is None:
        out_of_time = jnp.asarray(False)
    else:
        out_of_time = next_state.step >= params.max_steps
    truncated = jnp.logical_or(out_of_time, extinct)

    timestep = TimeStep(obs=get_obs(next_state, params),
                        reward=aux.gained,
                        terminated=aux.died,
                        truncated=truncated,
                        alive=aux.alive)
    return timestep, next_state


def step(key: jax.Array,
         state: EnvState,
         action: jax.Array,
         bank: EnvState,
         params: EnvParams) -> tuple[TimeStep, EnvState]:
    """One step, auto-resetting the world when the episode ends.

    Follows the gymnax convention on which observation comes back: ``reward``,
    ``terminated`` and ``alive`` describe the step that just happened, while
    ``obs`` is already the *next* episode's first observation when a reset
    fired. Rollout code therefore never branches on episode boundaries -- but
    it must use ``truncated`` to cut value bootstrapping, since ``obs`` no
    longer belongs to the trajectory that produced the reward.

    Extinction counts as truncation: unlike the evolutionary path, where a dead
    population simply earns zero for the rest of the episode, a stalled world
    here would go on consuming rollout budget while producing no transitions.
    """
    timestep, next_state = step_env(state, action, params)
    fresh_obs, fresh_state = reset(key, bank, params)

    reset_now = timestep.truncated
    next_state = jax.tree.map(
        lambda fresh, carried: jnp.where(reset_now, fresh, carried),
        fresh_state, next_state)
    obs = jax.tree.map(
        lambda fresh, carried: jnp.where(reset_now, fresh, carried),
        fresh_obs, timestep.obs)
    return timestep.replace(obs=obs), next_state
