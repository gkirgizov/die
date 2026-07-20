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

from die.jx.env import agent_cells, init_state, step_with_aux
from die.jx.state import (
    AGENT_ALIVE,
    AGENT_FOOD,
    MEDIUM_CHEM,
    MEDIUM_FOOD,
    EnvParams,
    EnvState,
)


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
    :param cells: ``[2, max_agents]`` int32 grid indices, one per agent. Carried
        so the observation is self-sufficient: a convolutional policy produces a
        field and each agent has to gather from it at its own position, and
        rederiving that outside would mean handing the policy raw state.
    """

    medium: jax.Array
    agent: jax.Array
    cells: jax.Array


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
    ix, iy = agent_cells(state.agents, params)
    return Obs(medium=state.medium,
               agent=state.agents[jnp.array([AGENT_FOOD, AGENT_ALIVE])],
               cells=jnp.stack([ix, iy]))


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


def perturb_medium(key: jax.Array, medium: jax.Array, params: EnvParams) -> jax.Array:
    """Per-step noise on the food and chemical channels.

    Applied here rather than in :mod:`die.jx.env` so the physics core stays
    RNG-free and bit-comparable with the xarray engine -- AGENTS.md's "two
    engines, one physics" invariant. The cost is that this perturbation has no
    xarray twin, so a policy trained under noise *replays* in a clean world.

    The two channels are perturbed **differently**, and the asymmetry is the
    whole point:

    *Food is multiplicative.* It has to be mean-preserving, because food is
    reward. Additive noise clipped at zero is a rectifier: on an empty cell it
    turns negative draws into 0 and positive draws into free resource, so at
    sigma=0.05 an empty cell gains +0.02 food per step out of nothing. Measured,
    after it inverted an entire robustness experiment -- policies scored *higher*
    under "noise" because the noise was feeding them. Multiplicative noise
    perturbs the level of resource that is already there and keeps empty cells
    empty.

    **``food_noise`` is only meaningful up to about 0.5.** The clip at zero
    reintroduces the same upward bias once negative draws become common:
    ``E[max(1 + s*Z, 0)] = Phi(1/s) + s*phi(1/s)``, which is +0.0% at s=0.25,
    +0.4% at s=0.5, +1.2% at s=0.6 and +8.3% at s=1.0. Above ~0.6 the "harder"
    world is measurably a richer one and results stop being interpretable.

    *Chemical is additive.* Chem is not reward, it is signal, and a corrupted
    communication channel must be able to produce **false positives** --
    multiplicative noise could only ever attenuate a message that was already
    there, never fabricate one. The resulting positive offset is a constant
    haze that observation normalization absorbs, leaving the variance, which is
    exactly the signal-to-noise knob intended: chem is the only route by which
    information reaches a distant agent, so this degrades cooperation
    specifically rather than perception in general.
    """
    if not params.stochastic_medium:
        return medium

    food_key, chem_key = jax.random.split(key)
    shape = medium[MEDIUM_FOOD].shape
    food = medium[MEDIUM_FOOD] * (1. + params.food_noise * jax.random.normal(food_key, shape))
    chem = medium[MEDIUM_CHEM] + params.comm_noise * jax.random.normal(chem_key, shape)
    return medium.at[MEDIUM_FOOD].set(jnp.maximum(food, 0.)) \
                 .at[MEDIUM_CHEM].set(jnp.maximum(chem, 0.))


def step_env(key: jax.Array,
             state: EnvState,
             action: jax.Array,
             params: EnvParams) -> tuple[TimeStep, EnvState]:
    """One step of raw physics, without auto-reset.

    ``action`` is ``[3, max_agents]`` -- (dx, dy, deposit1), already scaled into
    world units by the caller. ``key`` drives medium noise only, and is unused
    when ``stochastic_medium`` is off.
    """
    next_state, _, aux = step_with_aux(state, action, params)
    next_state = next_state.replace(
        medium=perturb_medium(key, next_state.medium, params))

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
    noise_key, reset_key = jax.random.split(key)
    timestep, next_state = step_env(noise_key, state, action, params)
    fresh_obs, fresh_state = reset(reset_key, bank, params)

    reset_now = timestep.truncated
    next_state = jax.tree.map(
        lambda fresh, carried: jnp.where(reset_now, fresh, carried),
        fresh_state, next_state)
    obs = jax.tree.map(
        lambda fresh, carried: jnp.where(reset_now, fresh, carried),
        fresh_obs, timestep.obs)
    return timestep.replace(obs=obs), next_state
