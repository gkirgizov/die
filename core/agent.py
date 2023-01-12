from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Union, List, Tuple, TypeVar, Callable, Sequence

import numpy as np
import xarray as da
import gymnasium as gym

from core import utils
from core.base_types import ActType, ObsType, MaskType, AgtType, MediumType
from core.data_init import DataInitializer


class Agent(ABC):
    @abstractmethod
    def forward(self, obs: ObsType) -> ActType:
        pass

    # @lru_cache(maxsize=1)
    # def coordgrid(self, field_size):
    #     return utils.get_meshgrid(field_size)

    @staticmethod
    def postprocess_action(agents: AgtType, action: ActType) -> ActType:
        masked = Agent._masked_alive(agents, action)
        rescaled = Agent._rescale_outputs(masked)
        return rescaled

    # TODO: rescaling: seems unnecessary because of [0, 1] bounds?
    @staticmethod
    def _rescale_outputs(action: ActType) -> ActType:
        """Scale outputs to their natural boundaries."""
        return action

    # TODO: agent masking: seems unnecessary because of natural Env logic?
    @staticmethod
    def _masked_alive(agents: AgtType, action: ActType) -> ActType:
        agent_alive_mask = agents.sel(channel='alive') > 0
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


class GradientAgent(Agent):
    """
    Research questions:
    - Passive selection by dying? vs. Active chemical trail depending on amount of food seen.
    """

    kinds = ('const', 'gaussian_noise')

    def __init__(self,
                 scale: float = 1.0,
                 deposit: float = 0.1,
                 inertia: float = 0.5,
                 kind: str = 'gaussian_noise',
                 noise_scale: float = 0.025,
                 ):
        if kind not in self.kinds:
            raise ValueError(f'Unknown kind of agent {kind}')
        self._rng = np.random.default_rng()
        self._kind = kind or self.kinds[0]
        self._noise_scale = noise_scale
        self._scale = scale
        self._deposit = deposit
        self._inertia = inertia
        self._grads_buffer = [0] * 2  # initially zeros

    def forward(self, obs: ObsType) -> ActType:
        coords = ['dx', 'dy']
        agents, medium = obs
        action = DataInitializer.init_action_for(medium)

        # Compute chemical gradient
        grads_buffer = []
        chemical = medium.sel(channel='chem1')
        grads = np.gradient(chemical)
        for coord, grad_i, prev_grad_i in zip(coords, grads, self._grads_buffer):
            # Add some noise
            if self._kind == 'gaussian_noise':
                noise = self._noise_scale * self._rng.normal(loc=0., scale=0.4, size=grad_i.shape)
            else:
                noise = 0
            # Compute gradient (i.e. direction) with inertia
            grad_val = (1 - self._inertia) * grad_i + self._inertia * prev_grad_i
            grads_buffer.append(grad_val)
            # Compute final value
            action.loc[dict(channel=coord)] = (grad_val + noise) * self._scale
        self._grads_buffer = grads_buffer

        # Chemical deposit relative to discovered food
        medium_food = medium.sel(channel='env_food')
        deposit = self._deposit * medium_food
        action.loc[dict(channel='deposit1')] = deposit

        # return self.postprocess_action(agents, action)
        return action


class ConstAgent(Agent):
    def __init__(self, delta_xy: Tuple[float, float], deposit: float = 0.):
        self._data = {'dx': delta_xy[0],
                      'dy': delta_xy[1],
                      'deposit1': deposit}

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        # at each agent location write our const vector
        action = DataInitializer.init_action_for(medium)
        for chan in action.coords['channel'].values:
            action.loc[dict(channel=chan)] = self._data[chan]

        return action
        # return self.postprocess_action(agents, action)


class RandomAgent(Agent):
    def __init__(self, move_scale: float = 0.1, deposit_scale: float = 0.5):
        self._scale = move_scale
        self._dep_scale = deposit_scale

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs

        s = self._scale
        action = DataInitializer.action_for(medium) \
            .with_noise('dx', -s, s) \
            .with_noise('dy', -s, s) \
            .with_noise('deposit1', 0., self._dep_scale) \
            .build()

        return action
        # return self.postprocess_action(agents, action)
