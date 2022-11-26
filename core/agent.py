from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Union, List, Tuple, TypeVar, Callable

import numpy as np
import xarray as da
import gymnasium as gym

from core.base_types import ActType, ObsType, MaskType, AgtType, MediumType
from core.data_init import DataInitializer


class Agent(ABC):
    @abstractmethod
    def forward(self, obs: ObsType) -> ActType:
        pass

    @staticmethod
    def postprocess_action(agents: AgtType, action: ActType) -> ActType:
        masked = Agent._masked_alive(agents, action)
        rescaled = Agent._rescale_outputs(masked)
        return rescaled

    # TODO: rescaling
    @staticmethod
    def _rescale_outputs(action: ActType) -> ActType:
        """Scale outputs to their natural boundaries."""
        return action

    @staticmethod
    def _masked_alive(agents: AgtType, action: ActType) -> ActType:
        agent_alive_mask = agents.sel(channel='agents') > 0
        masked_action = action * agent_alive_mask
        return masked_action


class ModelAgentSket(Agent):
    def __init__(self):
        # TODO: need joint channels info for the system:
        #  it must know about its own presence
        self.model: Callable[[ObsType], ActType] = None

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        action = self.model(obs)

        result_action = self.postprocess_action(agents, action)

        return result_action


class SimpleKernelAgent(Agent):
    def __init__(self, deposit: float = 0.1):
        self._deposit = deposit

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        coordinate_grid = Env
        action = self.model(obs)

        result_action = self.postprocess_action(agents, action)

        return result_action



class ConstAgent(Agent):
    def __init__(self, delta_xy: Tuple[float, float], deposit: float = 0.):
        self._data = {'dx': delta_xy[0],
                      'dy': delta_xy[1],
                      'deposit1': deposit}

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        # at each agent location write our const vector
        action = DataInitializer.init_action_for(agents)
        for chan in action.coords['channel'].values:
            action.loc[dict(channel=chan)] = self._data[chan]

        return self.postprocess_action(agents, action)


class RandomAgent(Agent):
    def __init__(self, move_scale: float = 0.1, deposit_scale: float = 0.5):
        self._scale = move_scale
        self._dep_scale = deposit_scale

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        s = self._scale
        action = DataInitializer.action_for(agents) \
            .with_noise('dx', -s, s) \
            .with_noise('dy', -s, s) \
            .with_noise('deposit1', 0., self._dep_scale) \
            .build()

        return self.postprocess_action(agents, action)
