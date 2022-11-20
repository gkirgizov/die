from typing import Optional, Union, List, Tuple, TypeVar

import numpy as np
import xarray as da
import gymnasium as gym

from core.base_types import ActType, ObsType, MaskType

RenderFrame = TypeVar('RenderFrame')


class Env(gym.Env[ObsType, ActType]):
    # TODO: maybe use Dataset with aligned `agents` and `medium` DataArrays
    #  with channels: x, y, food ??

    @staticmethod
    def _init_medium_array(field_size: Tuple[int, int]) -> ObsType:
        # TODO: init data
        # TODO: init agents
        channels = ('agents', 'agent-food', 'env-food', 'chem1')
        name = 'medium'

        shape = (*field_size, len(channels))
        xs = np.linspace(0, 1, field_size[0])
        ys = np.linspace(0, 1, field_size[1])

        # Parameters
        agents_ratio = 0.05  # ratio of cells with agents

        medium = da.DataArray(
            dims=('x', 'y', 'channels'),
            coords={'x': xs, 'y': ys, 'channels': channels},
            name=name,
        )
        return medium

    @staticmethod
    def _init_agential_array(field_size: Tuple[int, int]) -> ActType:
        channels = ('dist', 'turn', 'deposit1')
        name = 'agents'

        shape = (*field_size, len(channels))
        xs = np.linspace(0, 1, field_size[0])
        ys = np.linspace(0, 1, field_size[1])

        medium = da.DataArray(
            dims=('x', 'y', 'channels'),
            coords={'x': xs, 'y': ys, 'channels': channels},
            name=name,
        )
        return medium

    def __init__(self, field_size: Tuple[int, int]):
        self.medium = self._init_medium_array(field_size)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self._agent_move(action)
        self._agent_act_on_medium(action)

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
        pass

    def _agent_feed(self):
        """Gains food from environment"""
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
