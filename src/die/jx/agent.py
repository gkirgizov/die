"""An :class:`~die.agents.base.Agent` wrapper around the Flax NCA.

This is what lets JAX-trained kernels flow back into the rest of the project:
wearing the standard Agent interface, they run inside the xarray env through
the usual :class:`~die.sim.Simulation`, so the live view, the animation
recorder and ``die replay`` all work unchanged.

Caveat: replay therefore runs the *xarray* physics, which differs slightly from
the training physics -- collided deposits resolve differently (see
``tests/unit/test_jx_env.py::TestDepositCollisions``). Behaviorally equivalent,
not bit-equal.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from die.agents.base import Agent
from die.data_init import DataInitializer
from die.jx.model import ConvNCA, init_params
from die.jx.state import EnvConfig
from die.types import ActType, DataChannels, ObsType

#: Magic marker distinguishing these checkpoints from torch ones.
CHECKPOINT_FORMAT = 'die-jx-nca'
CHECKPOINT_VERSION = 1


class JaxNCAAgent(Agent):
    """Neural cellular automata agent whose parameters live in a flax pytree."""

    #: Trainer checkpoints use this suffix instead of torch's '.pt'.
    checkpoint_suffix = '.msgpack'

    def __init__(self,
                 params=None,
                 scale: float = 0.1,
                 deposit: float = 1.0,
                 kernel_sizes: Sequence[int] = (3, 3),
                 with_agent_channel: bool = True,
                 seed: int | None = None):
        self._init_params = dict(scale=scale, deposit=deposit,
                                 kernel_sizes=list(kernel_sizes),
                                 with_agent_channel=with_agent_channel)
        self.obs_channels = list(DataChannels.medium) \
            if with_agent_channel else list(DataChannels.medium[1:])
        self.model = ConvNCA(kernel_sizes=tuple(kernel_sizes),
                             obs_channels=len(self.obs_channels),
                             act_channels=len(DataChannels.actions))
        self.action_coefs = jnp.asarray([scale, scale, deposit], dtype=jnp.float32)
        self.params = params
        self._seed = seed
        self._act_field: np.ndarray | None = None

    @classmethod
    def create(cls,
               field_size: tuple[int, int],
               seed: int | None = None,
               **kwargs) -> JaxNCAAgent:
        """Builds an agent with freshly initialized parameters."""
        agent = cls(seed=seed, **kwargs)
        config = EnvConfig(field_size=field_size, max_agents=field_size[0] * field_size[1])
        key = jax.random.key(seed if seed is not None else 0)
        agent.params = init_params(agent.model, config, key)
        return agent

    def with_params(self, params) -> JaxNCAAgent:
        """Returns a copy carrying different parameters (same architecture)."""
        return JaxNCAAgent(params=params, seed=self._seed, **self._init_params)

    @property
    def init_params(self) -> dict[str, Any]:
        return self._init_params

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        sensed = jnp.asarray(medium.sel(channel=self.obs_channels).values, dtype=jnp.float32)
        act_field = self.model.apply(self.params, sensed)
        self._act_field = np.asarray(act_field)

        # Per-agent gather at nearest cells, mirroring AgentIndexer.
        width, height = medium.shape[1], medium.shape[2]
        coords = agents.sel(channel=['x', 'y']).values
        ix = np.clip(np.round(coords[0] * (width - 1)).astype(int), 0, width - 1)
        iy = np.clip(np.round(coords[1] * (height - 1)).astype(int), 0, height - 1)
        per_agent = self._act_field[:, ix, iy] * np.asarray(self.action_coefs)[:, None]

        return DataInitializer.init_action_for(agents, per_agent)

    def render(self) -> Sequence[np.ndarray | None]:
        if self._act_field is None:
            return [None]
        return [np.moveaxis(self._act_field, 0, -1)]

    def save(self, file: str | os.PathLike | io.FileIO):
        payload = {
            'format': CHECKPOINT_FORMAT,
            'version': CHECKPOINT_VERSION,
            'init_params': self._init_params,
            'params': serialization.to_state_dict(self.params),
        }
        data = serialization.msgpack_serialize(payload)
        if isinstance(file, io.FileIO):
            file.write(data)
        else:
            with open(file, 'wb') as handle:
                handle.write(data)

    @classmethod
    def load(cls, file: str | os.PathLike | io.FileIO) -> JaxNCAAgent:
        if isinstance(file, io.FileIO):
            data = file.read()
        else:
            with open(file, 'rb') as handle:
                data = handle.read()
        payload = serialization.msgpack_restore(data)

        fmt = payload.get('format')
        if fmt != CHECKPOINT_FORMAT:
            raise ValueError(f'Not a {CHECKPOINT_FORMAT} checkpoint (found format={fmt!r})')

        init_kwargs = dict(payload['init_params'])
        init_kwargs['kernel_sizes'] = [int(k) for k in init_kwargs['kernel_sizes']]
        agent = cls(**init_kwargs)
        # msgpack restores plain nested dicts of arrays -- exactly the pytree
        # shape flax produced, so it can be used as parameters directly.
        agent.params = jax.tree.map(jnp.asarray, payload['params'])
        return agent
