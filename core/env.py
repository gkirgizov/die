from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, TypeVar

import numpy as np
import xarray as da
import gymnasium as gym

from core.base_types import ActType, ObsType, MaskType, CostOperator, medium_channels, action_channels
from core.data_init import DataInitializer

RenderFrame = TypeVar('RenderFrame')

"""
Plan
- [x] some basic data init ENVs per channels for tests YET minimal
- [ ] test data init: plot channels
- [ ] plotting NB: all tests are isolated visual cases, really

- [ ] agent move
- [ ] MockConstAgent
- [ ] test agent moving

- [ ] test agent feeding & life cycle
- [ ] test medium diffusion
- [ ] test deposit with communication

- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

"""

# TODO:
@dataclass
class Dynamics:
    op_action_cost: CostOperator
    rate_feed: float


class Env(gym.Env[ObsType, ActType]):
    # TODO: maybe use Dataset with aligned `agents` and `medium` DataArrays
    #  with channels: x, y, food ??

    @staticmethod
    def init_medium_array(field_size: Tuple[int, int],
                          init_data: Optional[np.ndarray] = None) -> ObsType:
        channels = list(medium_channels)
        name = 'medium'

        shape = (*field_size, len(channels))
        xs = np.linspace(0, 1, field_size[0])
        ys = np.linspace(0, 1, field_size[1])

        # Parameters
        if init_data is None:
            init_data = DataInitializer(field_size)\
                .with_agents(ratio=0.05) \
                .with_food_perlin(threshold=0.1) \
                .build()

        medium = da.DataArray(
            data=init_data,
            dims=('channel', 'x', 'y'),
            coords={'x': xs, 'y': ys, 'channel': channels},
            name=name,
        )
        return medium

    @staticmethod
    def init_agential_array(field_size: Tuple[int, int]) -> ActType:
        channels = list(action_channels)
        name = 'agents'

        shape = (*field_size, len(channels))
        xs = np.linspace(0, 1, field_size[0])
        ys = np.linspace(0, 1, field_size[1])

        medium = da.DataArray(
            data=np.zeros(shape, dtype=np.float),
            dims=('channel', 'x', 'y'),
            coords={'x': xs, 'y': ys, 'channel': channels},
            name=name,
        )
        return medium

    def __init__(self, field_size: Tuple[int, int]):
        self.medium = self.init_medium_array(field_size)
        self.dynamics = Dynamics()  # TODO

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self._agent_move(action)
        self._agent_act_on_medium(action)
        self._agent_consume_stock(action)

        self._agent_feed()
        self._agent_lifecycle()

        self._medium_resource_dynamics()
        self._medium_diffuse()

        obs = self._agent_sense()

        # TODO: total reward? ending conditions? etc.

        return obs

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def _medium_diffuse(self):
        """Applies per-channel diffusion, channel-specific."""
        pass

    def _medium_resource_dynamics(self):
        """Defines agent-independent inflow & outflow of resource."""
        pass

    def _agent_sense(self) -> ObsType:
        """Apply neighbourhood mask."""
        mask = self._get_sense_mask()
        visible_medium = self._get_masked_medium(mask)
        return visible_medium

    def _agent_move(self, action: ActType):
        """Builds movement transformation for `agents` channels of the medium
        based on agents' provisioned action."""
        pass

    def _agent_act_on_medium(self, action: ActType):
        """Act on medium & consume required internal resource."""
        amount1 = action.sel(channel='deposit1')
        self.medium.loc[dict(channel='chem1')] += amount1

    def _agent_consume_stock(self, action: ActType):
        consumed = self.dynamics.op_action_cost(action)
        self.medium.loc[dict(channel='agent_food')] -= consumed

    def _agent_feed(self):
        """Gains food from environment"""
        # TODO:
        env_stock = self.medium.sel(channel='env_food')
        gained = self.dynamics.rate_feed * env_stock
        pass

    def _agent_lifecycle(self):
        """Dies if consumed all internal stock,
        grows if has excess stock & favorable conditions."""
        pass

    def _get_agent_mask(self) -> MaskType:
        pass

    def _get_sense_mask(self) -> MaskType:
        pass

    def _get_masked_medium(self, mask: MaskType) -> ObsType:
        pass


if __name__ == '__main__':
    pass
