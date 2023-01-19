from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Union, List, Tuple, TypeVar, Callable, Sequence

import numpy as np
from sklearn.preprocessing import normalize
import xarray as da
import gymnasium as gym

from core import utils
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
                 field_size: Tuple[int, int],
                 scale: float = 1.0,
                 deposit: float = 0.1,
                 inertia: float = 0.5,
                 kind: str = 'gaussian_noise',
                 noise_scale: float = 0.025,
                 normalized_grad: bool = False,
                 ):
        if kind not in self.kinds:
            raise ValueError(f'Unknown kind of agent {kind}')
        self._field_size = field_size
        self._rng = np.random.default_rng()
        self._kind = kind or self.kinds[0]
        self._noise_scale = noise_scale
        self._scale = scale
        self._deposit = deposit
        self._inertia = inertia
        self._normalized = normalized_grad
        self._prev_grad = self._get_some_noise()

    def _get_some_noise(self):
        ncoords = 2
        noise = self._noise_scale * self._rng.normal(loc=0., scale=0.4,
                                                     size=(ncoords, *self._field_size))
        return noise

    def _get_gradient(self, field) -> np.ndarray:
        # coordinate grads are grouped into the first axis through 'np.stack'
        grad = np.stack(np.gradient(field))
        if self._normalized:
            w, h = self._field_size
            # sklearn normalize works only with 2d arrays, that's why reshape
            flat_grad = grad.reshape((2, w * h))
            grad = normalize(flat_grad, axis=0, norm='l2').reshape((2, w, h))
        return grad

    def forward(self, obs: ObsType) -> ActType:
        coords = ['dx', 'dy']
        agents, medium = obs
        action = DataInitializer.init_action_for(medium)

        # Compute chemical gradient
        chemical = medium.sel(channel='chem1')
        grad = self._get_gradient(chemical)
        # Compute gradient (i.e. direction) with inertia
        grad = (1 - self._inertia) * grad + self._inertia * self._prev_grad
        # Compute final value with noise
        noise = self._get_some_noise() if self._kind == 'gaussian_noise' else 0
        action.loc[dict(channel=coords)] = (grad + noise) * self._scale
        self._prev_grad = grad

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
