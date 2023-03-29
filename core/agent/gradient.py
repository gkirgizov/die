from typing import Optional, Sequence

import numpy as np
import scipy
import xarray as da

from core.agent.base import Agent
from core.base_types import AgtType, ActType, ObsType
from core.data_init import DataInitializer
from core.utils import get_radians, polar2xy, AgentIndexer, discretize, renormalize_radians, xy2polar


class GradientAgent(Agent):
    """
    Research questions:
    - Passive selection by dying? vs. Active chemical trail depending on amount of food seen.
    """

    def __init__(self,
                 max_agents: int = 10**6,
                 scale: float = 0.01,
                 deposit: float = 4.0,
                 inertia: float = 0.9,
                 sense_offset: float = 0.,
                 noise_scale: float = 0.025,  # is measured as fraction of 'scale'
                 normalized_grad: bool = True,
                 grad_clip: Optional[float] = 1e-5,
                 ):
        self._size = max_agents
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

        self._render_grad = None

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

    def _process_deposit(self, agents: AgtType, sensed_food: AgtType) -> ActType:
        return self._deposit * sensed_food

    def forward(self, obs: ObsType) -> ActType:
        coords = ['dx', 'dy']
        agents, medium = obs
        action = DataInitializer.init_action_for(agents)
        idx = AgentIndexer(medium.shape[1:], agents)
        chemical = medium.sel(channel='chem1')

        # Compute chemical gradient with custom processing
        grad_field = self._get_gradient(chemical)
        grad_per_agent = idx.field_by_agents(grad_field, only_alive=False, offset=self._sense_offset)
        grad_per_agent = self._process_gradient(grad_per_agent)
        grad_per_agent = self._process_momentum(grad_per_agent)

        # Update agent direction after all transformations
        self._direction_rads = get_radians(grad_per_agent)
        # Update render data
        self._render_grad = grad_field

        # Chemical deposit relative to discovered food
        food = medium.sel(channel='env_food')
        sensed_medium_food = idx.field_by_agents(food, only_alive=False)
        deposit = self._process_deposit(agents, sensed_medium_food)

        # Assign action
        action.loc[dict(channel=coords)] = grad_per_agent * self._scale
        action.loc[dict(channel='deposit1')] = deposit

        # return self.postprocess_action(agents, action)
        return action

    def render(self) -> Sequence[np.ndarray]:
        if self._render_grad is None:
            pixel = np.ones((1, 1, 3))
            return [pixel]
        r = self._render_grad.sel(channel='dx').values
        g = self._render_grad.sel(channel='dy').values
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)
        rgb = 0.5 * (rgb + 1.)  # rescale from [-1, 1] to [0, 1]
        return [rgb]


class PhysarumAgent(GradientAgent):
    def __init__(self,
                 max_agents: int = 10**6,
                 scale: float = 0.005,
                 deposit: float = 4.0,
                 inertia: float = 0.0,
                 sense_offset: float = 0.03,
                 noise_scale: float = 0.0,
                 normalized_grad: bool = True,
                 grad_clip: Optional[float] = 1e-5,
                 turn_angle: int = 30,
                 sense_angle: int = 90,
                 turn_tolerance: float = 0.1,  # relative (to turn angle) tolerance for definite turn
                 ):
        super().__init__(max_agents,
                         scale, deposit, inertia,
                         sense_offset,
                         noise_scale,
                         normalized_grad, grad_clip)
        self._turn_radians = np.radians(turn_angle)
        self._sense_radians = np.radians(sense_angle)
        self._rtol = turn_tolerance
        self._direction_rads = self._discretize_grad(self._prev_grad)
        self._deposit_mask = 1.

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
        dir_delta *= np.logical_not(undetermined)
        turn = rand_choice
        turn[dir_delta > atol] = -1  # right, clockwise
        turn[dir_delta < -atol] = 1 # left, counter-clockwise
        turn *= self._turn_radians

        # Create deposit mask so agents signal chemical only on succesful turn
        self._deposit_mask = np.logical_not(undetermined_grad | undetermined_turn)

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

    def _process_deposit(self, agents: AgtType, sensed_food: AgtType) -> ActType:
        mask = self._deposit_mask.clip(0.1, 1.0)  # allow minimal signaling always
        target = sensed_food
        # target = agents.sel(channel='agent_food')
        return self._deposit * target * mask

    def _process_gradient(self, grad: np.ndarray) -> np.ndarray:
        delta_grad = self._discrete_turn(grad)
        # delta_grad = np.stack(polar2xy(1., self._direction_rads))  # const direction
        return delta_grad
