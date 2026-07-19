"""World state and static configuration of the JAX env.

The split matters for jit: :class:`EnvState` is a pytree that flows *through*
traced code, while :class:`EnvConfig` is static (frozen, hashable) and is baked
into the compiled function via closure. Changing a config value therefore
triggers a recompile; changing state does not.
"""

from __future__ import annotations

import dataclasses

import jax
from flax import struct

# Channel orders mirror die.types.DataChannels -- kept as plain indices here
# because JAX arrays carry no channel labels.
MEDIUM_AGENTS, MEDIUM_FOOD, MEDIUM_CHEM = 0, 1, 2
AGENT_X, AGENT_Y, AGENT_ALIVE, AGENT_FOOD = 0, 1, 2, 3
ACT_DX, ACT_DY, ACT_DEPOSIT = 0, 1, 2


@struct.dataclass
class EnvState:
    """Fixed-shape world state.

    :param medium: ``[3, width, height]`` -- channels (agents, env_food, chem1).
    :param agents: ``[4, max_agents]`` -- channels (x, y, alive, agent_food).
        Dead slots are parked at (0, 0) with ``alive=0``; every per-agent
        computation must mask by ``alive`` (same invariant as the xarray env).
    :param step: step counter, drives time-dependent food flow.
    """

    medium: jax.Array
    agents: jax.Array
    step: jax.Array


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """Static dynamics parameters -- the JAX twin of :class:`die.env.Dynamics`.

    Operators that are callables in ``Dynamics`` become string tags here so the
    config stays hashable and jit-cacheable.
    """

    field_size: tuple[int, int]
    max_agents: int
    rate_feed: float = 0.1
    rate_decay_chem: float = 0.1
    diffuse_sigma: float = 0.5
    boundary: str = 'wrap'          # 'wrap' | 'limit'
    #: weights of linear_action_cost: (deposit, distance)
    cost_weights: tuple[float, float] = (0.02, 0.01)
    food_infinite: bool = False
    agents_die: bool = False
    #: 'none' -> identity flow; 'wave' -> WaveSequence flow (dyn-pred)
    food_flow: str = 'none'
    wave_scale: float = 0.5
    wave_decay: float = 0.5
    wave_dt: float = 0.01
    wave_t_bounds: tuple[float, float] = (0.0, 10.0)
    init_agent_ratio: float = 0.1

    @property
    def width(self) -> int:
        return self.field_size[0]

    @property
    def height(self) -> int:
        return self.field_size[1]
