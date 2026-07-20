"""One interface over every agent in the repo, scripted or learned.

The repo's :class:`~die.agents.base.Agent` interface is xarray-shaped: it takes
a ``(agents, medium)`` DataArray pair and returns a DataArray action. That is
the right interface for the live view and for ``die replay``, and the wrong one
for a benchmark -- it cannot be jitted, scanned or vmapped, so scoring 19
scenarios x 6 seeds x 8 agents through it would run a Python loop over every
world step.

So learned checkpoints are adapted here to the same functional
:class:`~die.bench.runner.Policy` protocol the scripted JAX ports already
implement. The weights are identical to what ``die replay`` would load; only
the plumbing differs.

Both learned policies act **deterministically**, on the distribution mean. That
matches how the RL code evaluates during training, and it is what makes a
single-rollout score meaningful rather than a sample from a policy's noise.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct

from die.jx.agent import JaxNCAAgent
from die.jx.interface import Obs
from die.jx.state import EnvParams
from die.rl.agent import RLAgent
from die.rl.networks import squash


@struct.dataclass
class NoState:
    """Carry for stateless policies. Both learned agents are pure functions of
    the current observation -- unlike the scripted gradient followers, which
    integrate direction over time."""


class NCAPolicy:
    """An evolved :class:`~die.jx.agent.JaxNCAAgent` as a benchmark policy.

    Mirrors :func:`die.jx.model.compute_action`: the model reads the whole
    medium once and every agent gathers the resulting action field at its own
    cell, including dead slots, which the env masks out downstream.
    """

    def __init__(self, agent: JaxNCAAgent):
        self._model = agent.model
        self._params = agent.params
        self._coefs = agent.action_coefs

    def init(self, key: jax.Array, params: EnvParams) -> NoState:
        return NoState()

    def act(self, key: jax.Array, state: NoState, obs: Obs, params: EnvParams):
        action_field = self._model.apply(self._params, obs.medium)
        ix, iy = obs.cells[0], obs.cells[1]
        return action_field[:, ix, iy] * self._coefs[:, None], state


class PPOPolicy:
    """A PPO-trained :class:`~die.rl.agent.RLAgent` as a benchmark policy.

    Observation normalization is applied with the statistics frozen into the
    checkpoint. Skipping it would shift every activation -- the network never
    saw raw medium values -- so the normalizer is as much part of the policy as
    the weights are.
    """

    def __init__(self, agent: RLAgent):
        self._network = agent.network
        self._params = agent.params
        self._normalizer = agent.normalizer
        init = agent.init_params
        self._normalize = init['normalize_obs']
        self._scales = jnp.asarray([init['action_scale'],
                                    init['action_scale'],
                                    init['deposit_scale']], dtype=jnp.float32)

    def init(self, key: jax.Array, params: EnvParams) -> NoState:
        return NoState()

    def act(self, key: jax.Array, state: NoState, obs: Obs, params: EnvParams):
        if self._normalize and self._normalizer is not None:
            obs = self._normalizer.apply(obs)
        mean, _, _ = self._network.apply(self._params, obs)
        # Network emits [max_agents, 3] in [-1, 1]; the env consumes
        # [3, max_agents] in world units.
        return squash(mean).T * self._scales[:, None], state


def load_learned(path: str | Path) -> NCAPolicy | PPOPolicy:
    """Builds a policy from a checkpoint, dispatching on its format marker.

    Mirrors :func:`die.registry.load_agent`'s sniffing rather than trusting the
    filename, so a renamed checkpoint still loads as the right kind.
    """
    path = Path(path)
    try:
        return PPOPolicy(RLAgent.load(path))
    except ValueError:
        return NCAPolicy(JaxNCAAgent.load(path))
