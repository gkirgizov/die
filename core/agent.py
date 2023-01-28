from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Union

import numpy as np
import scipy
import xarray as da

from core.base_types import ActType, ObsType, AgtType
from core.data_init import DataInitializer
from core.utils import xy2polar, polar2xy, renormalize_radians, discretize, AgentIndexer, get_radians


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

    def __init__(self,
                 num_agents: int,
                 scale: float = 1.0,
                 deposit: float = 0.1,
                 inertia: float = 0.5,
                 sense_offset: float = 0.,
                 noise_scale: float = 0.025,  # is measured as fraction of 'scale'
                 normalized_grad: bool = True,
                 grad_clip: Optional[float] = 1e-5,
                 ):
        self._size = num_agents
        self._rng = np.random.default_rng()
        self._noise_scale = noise_scale
        self._scale = scale
        self._deposit = deposit
        self._inertia = inertia
        self._sense_offset_scale = sense_offset
        self._normalized = normalized_grad
        self._grad_clip = grad_clip

        self._prev_grad = self._get_some_noise()
        self._direction_rads = get_radians(self._prev_grad)

    def _get_some_noise(self):
        ncoords = 2
        noise = self._rng.normal(loc=0., scale=0.4, size=(ncoords, self._size))
        return noise

    def _get_gradient(self, field: da.DataArray) -> da.DataArray:
        # coordinate grads are grouped into the first axis through 'np.stack'
        grad = np.stack(np.gradient(field))

        norm = scipy.linalg.norm(grad, axis=0, ord=2)
        if self._normalized:
            with np.errstate(divide='ignore', invalid='ignore'):
                grad = np.nan_to_num(np.true_divide(grad, norm))
        if self._grad_clip is not None:
            # Apply mask for too small gradients
            grad *= (norm >= self._grad_clip)

        return da.DataArray(data=grad,
                            coords={'channel': ['dx', 'dy'],
                                    'x': field.coords['x'],
                                    'y': field.coords['y'],
                                    })

    @property
    def _sense_offset(self) -> np.ndarray:
        offset_xy = np.stack(polar2xy(self._sense_offset_scale, self._direction_rads))
        return offset_xy

    def _process_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Designed for custom processing in gradient agent subclasses."""
        return grad

    def _process_momentum(self, grad: np.ndarray) -> np.ndarray:
        """Process gradient by adding inertia and noise"""
        # Compute gradient (i.e. direction) with inertia
        grad = (1 - self._inertia) * grad + self._inertia * self._prev_grad
        # Compute final value with noise
        noise = self._get_some_noise()
        grad += self._noise_scale * noise
        # Store grad for the next computation
        self._prev_grad = grad
        return grad

    def forward(self, obs: ObsType) -> ActType:
        coords = ['dx', 'dy']
        agents, medium = obs
        action = DataInitializer.init_action_for(agents)
        idx = AgentIndexer(agents)
        chemical = medium.sel(channel='chem1')

        # Compute chemical gradient with custom processing
        grad_field = self._get_gradient(chemical)
        grad_per_agent = idx.field_by_agents(grad_field, only_alive=False, offset=self._sense_offset)
        grad_per_agent = self._process_gradient(grad_per_agent)
        grad_per_agent = self._process_momentum(grad_per_agent)

        # Update agent direction after all transformations
        self._direction_rads = get_radians(grad_per_agent)

        # Chemical deposit relative to discovered food
        food = medium.sel(channel='env_food')
        sensed_medium_food = idx.field_by_agents(food, only_alive=False)
        deposit = self._deposit * sensed_medium_food

        # Assign action
        action.loc[dict(channel=coords)] = grad_per_agent * self._scale
        action.loc[dict(channel='deposit1')] = deposit

        # return self.postprocess_action(agents, action)
        return action


class PhysarumAgent(GradientAgent):
    def __init__(self,
                 num_agents: int,
                 scale: float = 0.01,
                 deposit: float = 0.1,
                 inertia: float = 0.5,
                 sense_offset: float = 0.,
                 noise_scale: float = 0.025,
                 normalized_grad: bool = True,
                 grad_clip: Optional[float] = 1e-5,
                 turn_angle: int = 30,
                 sense_angle: int = 90,
                 turn_tolerance: float = 0.1,  # relative (to turn angle) tolerance for definite turn
                 ):
        super().__init__(num_agents,
                         scale, deposit, inertia,
                         sense_offset,
                         noise_scale,
                         normalized_grad, grad_clip)
        self._turn_radians = np.radians(turn_angle)
        self._sense_radians = np.radians(sense_angle)
        self._rtol = turn_tolerance
        self._direction_rads = self._discretize_grad(self._prev_grad)

    def _discretize_grad(self, grad):
        return discretize(get_radians(grad), self._turn_radians)

    def _choose_turn(self, drads: np.ndarray) -> np.ndarray:
        """Chooses a turn based on delta between desired & actual direction"""
        # Compute delta between actual direction & chemical gradient direction
        dir_delta = self._direction_rads - drads
        dir_delta = renormalize_radians(dir_delta)
        atol = self._turn_radians * self._rtol

        # Random turn for indeterminate gradients
        undetermined_grad = np.isclose(0, drads, rtol=1e-5)
        undetermined_turn = np.isclose(0, dir_delta, rtol=1e-2, atol=atol)
        unseen_grad = abs(dir_delta) > self._sense_radians
        undetermined = undetermined_grad | undetermined_turn | unseen_grad
        rand_choice = (np.random.randint(0, 2, undetermined.shape) - 0.5) * 2

        # Turn right or left, except for undetermined values where random turn is made
        turn = rand_choice
        turn[dir_delta > atol] = -1  # right, clockwise
        turn[dir_delta < -atol] = 1 # left, counter-clockwise
        turn[undetermined] = rand_choice[undetermined]
        turn *= self._turn_radians

        return turn

    def _discrete_turn(self, grad: np.ndarray) -> np.ndarray:
        # Convert dx, dy to radians
        dx, dy = grad
        dr, drads = xy2polar(dx, dy)  # radian values are already normalized in (-pi, pi]

        # Make discrete movement choice
        turn_radians = self._choose_turn(drads)
        # Compute new direction
        directions = renormalize_radians(self._direction_rads + turn_radians)

        # Convert back from radians to (dx, dy)
        dr = 1. if self._normalized else dr
        grad = np.stack(polar2xy(dr, directions))
        return grad

    def _process_gradient(self, grad: np.ndarray) -> np.ndarray:
        delta_grad = self._discrete_turn(grad)
        # delta_grad = np.stack(polar2xy(1., self._direction_rads))  # const direction
        return delta_grad


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

        return action
        # return self.postprocess_action(agents, action)


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
            .build_agents()

        return action
        # return self.postprocess_action(agents, action)
