"""Episode rollout and batched population evaluation.

This is where the JAX port pays off: one jitted, ``vmap``-ed call evaluates a
whole population of candidate agents over a full episode, instead of a Python
loop of per-candidate simulations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from die.jx.env import step
from die.jx.model import ConvNCA, compute_action
from die.jx.state import EnvConfig, EnvState


def make_episode_return(model: ConvNCA,
                        config: EnvConfig,
                        num_steps: int,
                        action_coefs: jax.Array):
    """Builds ``(params, initial_state) -> total_reward``.

    ``model``, ``config`` and ``num_steps`` are captured statics, so the
    returned function recompiles only when they change -- the initial state
    flows through as a value and may vary per generation for free.
    """

    def episode_return(params, initial_state: EnvState) -> jax.Array:
        def scan_step(state, _):
            action = compute_action(model, params, state, config, action_coefs)
            state, reward = step(state, action, config)
            return state, reward

        _, rewards = lax.scan(scan_step, initial_state, length=num_steps)
        return rewards.sum()

    return episode_return


def make_evaluate_population(model: ConvNCA,
                             config: EnvConfig,
                             num_steps: int,
                             action_coefs: jax.Array):
    """Builds ``(population_params, initial_state) -> fitnesses``.

    The population axis is mapped over the parameter pytree while every
    candidate shares the same initial world -- a fair within-generation
    comparison, matching the trainer's per-generation eval seed.
    """
    episode_return = make_episode_return(model, config, num_steps, action_coefs)
    return jax.jit(jax.vmap(episode_return, in_axes=(0, None)))


def stack_params(params_list):
    """Stacks a list of parameter pytrees along a leading population axis."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *params_list)
