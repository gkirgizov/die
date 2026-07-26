"""An :class:`~die.agents.base.Agent` wrapper around a trained PPO policy.

The RL twin of :class:`~die.jx.agent.JaxNCAAgent`, and the same bargain: wearing
the standard Agent interface, a trained policy runs inside the xarray env
through the usual :class:`~die.sim.Simulation`, so the live view, the animation
recorder and ``die replay`` all work unchanged.

Two things ride in the checkpoint beyond network parameters, because the policy
is not deployable without them:

- **Observation statistics.** The network was trained on normalized inputs;
  feeding it raw medium values would shift every activation. The final running
  statistics are frozen into the checkpoint and applied at replay.
- **Action scales.** The policy emits [-1, 1]; the world consumes scaled units.

Replay acts on the distribution *mean* -- deterministic, matching how the
policy is evaluated during training. And as with the NCA agent, replay runs the
xarray physics: behaviorally equivalent to the training physics, not bit-equal,
and a policy trained under medium noise replays in a clean world.
"""

from __future__ import annotations

import io
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from die.agents.base import Agent
from die.data_init import DataInitializer
from die.jx.interface import Obs
from die.rl.networks import ActorCritic, squash
from die.rl.obs import ObsNormalizer, RunningStats
from die.types import ActType, ObsType

#: Magic marker distinguishing these checkpoints from 'die-jx-nca' ones.
CHECKPOINT_FORMAT = 'die-rl-ppo'
CHECKPOINT_VERSION = 1


class RLAgent(Agent):
    """A PPO-trained policy behind the standard Agent interface."""

    checkpoint_suffix = '.msgpack'

    def __init__(self,
                 params=None,
                 normalizer: ObsNormalizer | None = None,
                 features: int = 32,
                 num_blocks: int = 2,
                 kernel_size: int = 3,
                 hidden: int = 64,
                 action_scale: float = 0.1,
                 deposit_scale: float = 1.0,
                 normalize_obs: bool = True):
        self._init_params = dict(features=features, num_blocks=num_blocks,
                                 kernel_size=kernel_size, hidden=hidden,
                                 action_scale=action_scale,
                                 deposit_scale=deposit_scale,
                                 normalize_obs=normalize_obs)
        self.network = ActorCritic(features=features, num_blocks=num_blocks,
                                   kernel_size=kernel_size, hidden=hidden)
        self.params = params
        self.normalizer = normalizer

    @property
    def init_params(self) -> dict[str, Any]:
        return self._init_params

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        width, height = medium.shape[1], medium.shape[2]

        coords = agents.sel(channel=['x', 'y']).to_numpy()
        ix = np.clip(np.round(coords[0] * (width - 1)).astype(np.int32), 0, width - 1)
        iy = np.clip(np.round(coords[1] * (height - 1)).astype(np.int32), 0, height - 1)

        jx_obs = Obs(
            medium=jnp.asarray(medium.values, dtype=jnp.float32),
            agent=jnp.asarray(agents.sel(channel=['agent_food', 'alive']).values,
                              dtype=jnp.float32),
            cells=jnp.stack([jnp.asarray(ix), jnp.asarray(iy)]))
        if self._init_params['normalize_obs'] and self.normalizer is not None:
            jx_obs = self.normalizer.apply(jx_obs)

        mean, _, _ = self.network.apply(self.params, jx_obs)
        scales = np.asarray([self._init_params['action_scale'],
                             self._init_params['action_scale'],
                             self._init_params['deposit_scale']], dtype=np.float32)
        per_agent = np.asarray(squash(mean)).T * scales[:, None]
        return DataInitializer.init_action_for(agents, per_agent)

    def save(self, file: str | os.PathLike | io.FileIO):
        payload = {
            'format': CHECKPOINT_FORMAT,
            'version': CHECKPOINT_VERSION,
            'init_params': self._init_params,
            'params': serialization.to_state_dict(self.params),
            'normalizer': serialization.to_state_dict(self.normalizer)
                          if self.normalizer is not None else {},
        }
        data = serialization.msgpack_serialize(payload)
        if isinstance(file, io.FileIO):
            file.write(data)
        else:
            with open(file, 'wb') as handle:
                handle.write(data)

    @classmethod
    def load(cls, file: str | os.PathLike | io.FileIO) -> RLAgent:
        if isinstance(file, io.FileIO):
            data = file.read()
        else:
            with open(file, 'rb') as handle:
                data = handle.read()
        payload = serialization.msgpack_restore(data)

        fmt = payload.get('format')
        if fmt != CHECKPOINT_FORMAT:
            raise ValueError(f'Not a {CHECKPOINT_FORMAT} checkpoint (found format={fmt!r})')

        init_params = {key: (bool(value) if key == 'normalize_obs' else
                             int(value) if key in ('features', 'num_blocks',
                                                   'kernel_size', 'hidden') else
                             float(value))
                       for key, value in payload['init_params'].items()}
        agent = cls(**init_params)
        agent.params = jax.tree.map(jnp.asarray, payload['params'])
        if payload['normalizer']:
            stats = jax.tree.map(jnp.asarray, payload['normalizer'])
            agent.normalizer = ObsNormalizer(
                medium=RunningStats(**stats['medium']),
                agent=RunningStats(**stats['agent']))
        return agent
