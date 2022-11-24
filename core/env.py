from dataclasses import dataclass
from numbers import Number
from typing import Optional, Union, List, Tuple, TypeVar, Sequence, Hashable

import numpy as np
import xarray as da
import gymnasium as gym

from core.base_types import DataChannels, ActType, ObsType, MaskType, CostOperator, AgtType, MediumType
from core.data_init import DataInitializer

RenderFrame = TypeVar('RenderFrame')

"""
Plan
- [x] some basic data init ENVs per channels for tests YET minimal
- [x] test data init: plot channels
- [ ] plotting NB: all tests are isolated visual cases, really
      maybe with some statistical tests *over the image*
- [x] make buffered update

- [x] agent move async
- [x] MockConstAgent
- [ ] test agent moving

- [ ] agent lifecycle
- [ ] medium boundary conditions
- [ ] ...
- [ ] test agent feeding & life cycle
- [ ] test medium diffusion
- [ ] test deposit with communication

- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

"""


@dataclass
class Dynamics:
    op_action_cost: CostOperator
    rate_feed: float  # TODO: maybe do lambda taking into account input?

    @staticmethod
    def default_cost(action: ActType):
        dist = np.linalg.norm(action.sel(channel=['dx', 'dy']), axis=0)
        cost = 0.5 * action.sel(channel='deposit1') + 0.25 * dist
        return cost


class Env(gym.Env[ObsType, ActType]):
    # TODO: maybe use Dataset with aligned `agents` and `medium` DataArrays
    #  with channels: x, y, food ??
    def __init__(self, field_size: Tuple[int, int]):
        self.medium = DataInitializer(field_size, DataChannels.medium) \
            .with_food_perlin(threshold=0.1) \
            .build(name='medium')

        self.agents = DataInitializer(field_size, DataChannels.agents) \
            .with_agents(ratio=0.05) \
            .build(name='agents')

        # self.actions = DataInitializer(field_size, DataChannels.actions) \
        #     .build(name='actions')

        self.buffer_agents = self.agents.copy(deep=True)
        self.buffer_medium = self.medium.copy(deep=True)

        self.dynamics = Dynamics(op_action_cost=Dynamics.default_cost,
                                 rate_feed=0.1)

    def _rotate_agent_buffer(self, reset_data=0.):
        tmp = self.agents
        self.agents = self.buffer_agents
        self.buffer_agents = tmp
        self.buffer_agents = reset_data

    def _rotate_medium_buffer(self, reset_data=0.):
        tmp = self.medium
        self.medium = self.buffer_medium
        self.buffer_medium = tmp
        self.buffer_medium = reset_data

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Each substep here leaves the world state valid and ready for the next substep."""

        self._agent_feed()
        self._agent_act_on_medium(action)
        self._agent_consume_stock(action)
        self._agent_lifecycle()

        self._medium_resource_dynamics()
        self._medium_diffuse()

        # NB: only agents that are alive (after lifecycle step) will move
        self._agent_move_async(action)

        obs = self._get_sensed_medium

        # TODO: total reward? ending conditions? etc.
        reward = 0.

        return (self.agents, obs), reward, False, False, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def _medium_diffuse(self):
        """Applies per-channel diffusion, channel-specific."""
        pass

    def _medium_resource_dynamics(self):
        """Defines agent-independent inflow & outflow of resource."""
        pass

    @property
    def _get_agent_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_agent_mask.values.nonzero()

    def _iter_agents(self, shuffle: bool = True) -> Sequence[Tuple[int, int]]:
        xs, ys = self._get_agent_indices
        agent_inds = list(zip(xs, ys))
        if shuffle:
            np.random.default_rng().shuffle(agent_inds)  # in-place
        return agent_inds

    def _agent_move_async(self, action: ActType):
        for ix, iy in self._iter_agents(shuffle=True):
            ixy = dict(x=ix, y=iy)
            # Get all medium & movement data
            agent_action = action[ixy]
            dx, dy, deposit = agent_action.values

            # Get actual agent coords (from just indices)
            x = agent_action.coords['x'].values
            y = agent_action.coords['y'].values
            # Compute new pos
            cxy = dict(x=x + dx, y=y + dy)

            # Update agents
            self.buffer_agents.loc[cxy] = self.agents[ixy]  # move agent data to new position

            # Update medium by action: Deposit chemical trace
            # old_medium_data = self.medium[ixy]
            # TODO: do I deposit trace on old locations or new locations?
            # self.buffer_medium.loc[cxy].loc[dict(channel='chem1')] += deposit
        self._rotate_agent_buffer()

    def _agent_act_on_medium(self, action: ActType):
        """Act on medium & consume required internal resource."""
        amount1 = action.sel(channel='deposit1')
        self.medium.loc[dict(channel='chem1')] += amount1

    def _agent_consume_stock(self, action: ActType):
        consumed = self.dynamics.op_action_cost(action)
        self.agents.loc[dict(channel='agent_food')] -= consumed

    def _agent_feed(self):
        """Gains food from environment"""
        env_stock = self.medium.sel(channel='env_food')
        # TODO: what's best gain?
        gained = self.dynamics.rate_feed * env_stock
        # TODO: maybe coord-aligned summation will do that for me?
        self.agents.loc[dict(channel='agent_food')] += gained
        self.medium.loc[dict(channel='env_food')] -= gained

    def _agent_lifecycle(self):
        """Dies if consumed all internal stock,
        grows if has excess stock & favorable conditions."""
        pass

    @property
    def _get_agent_mask(self) -> MaskType:
        return self.agents.sel(channel='agents') > 0

    @property
    def _get_sense_mask(self) -> MaskType:
        pass

    @property
    def _get_agents_medium(self) -> MediumType:
        return self.medium.where(self._get_agent_mask)

    @property
    def _get_sensed_medium(self) -> MediumType:
        """Apply agent neighbourhood mask to the medium to get "what agents see"."""
        visible_medium = self.medium.where(self._get_sense_mask)
        return visible_medium

    def _meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xc = self.medium.coords['x'].values
        yc = self.medium.coords['y'].values
        return np.meshgrid(xc, yc)


if __name__ == '__main__':
    pass
