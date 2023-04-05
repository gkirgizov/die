import logging
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Optional, Union, List, Tuple, TypeVar, Sequence, Hashable, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import filters
import xarray as da
import gymnasium as gym

from core.base_types import DataChannels, ActType, ObsType, MaskType, CostOperator, AgtType, MediumType, FoodOperator, \
    Array1C, ActionFunc
from core.data_init import DataInitializer
from core import utils
from core.render import EnvRenderer
from core.utils import AgentIndexer, ChannelLogger

RenderFrame = TypeVar('RenderFrame')


class BoundaryCondition(Enum):
    wrap = 'wrap'
    limit = 'limit'


def linear_action_cost(action: ActType, weights=(0.02, 0.01)) -> Array1C:
    # Default cost of actions is computed as linear combination of:
    #  total crossed distance plus chemical deposition
    dist = np.linalg.norm(action.sel(channel=['dx', 'dy']), axis=0)
    deposit = np.abs(action.sel(channel='deposit1'))
    cost = weights[0] * deposit + weights[1] * dist
    return cost


def zero_cost(action: ActType) -> Array1C:
    return np.zeros(action.shape[1:])


@dataclass
class Dynamics:
    op_action_cost: CostOperator = linear_action_cost
    op_food_flow: FoodOperator = lambda x: x
    rate_feed: float = 0.1
    rate_decay_chem: float = 0.1
    boundary: BoundaryCondition = BoundaryCondition.wrap
    diffuse_mode: str = 'wrap'
    diffuse_sigma: float = .5

    # test options?
    apply_sense_mask: bool = False  # TODO: align sense mask with agents offsets
    # TODO: strict_cost
    strict_cost: bool = True  # defines if actions with cost higher than available resource are truncated
    food_infinite: bool = False
    agents_die: bool = False
    agents_born: bool = False

    # Initialization options
    init_agent_ratio: float = 0.1


class Env(gym.Env[ObsType, ActType]):
    def __init__(self,
                 field_size: Tuple[int, int],
                 dynamics: Optional[Dynamics] = None):
        self._field_size = field_size
        self.coordgrid = utils.get_meshgrid(field_size)
        self.dynamics = dynamics or Dynamics()
        self._renderer = EnvRenderer(field_size, field_colors_id='rgb')
        self._init_data(field_size)

    def _init_data(self, field_size: Tuple[int, int]):
        self.medium = DataInitializer(field_size, DataChannels.medium) \
            .with_const('env_food', 0.5) \
            .with_food_perlin(threshold=1.0, octaves=8) \
            .with_agents(ratio=self.dynamics.init_agent_ratio) \
            .build(name='medium')
        self.buffer_medium = self.medium.copy(deep=True)

        self.agents = DataInitializer.agents_from_medium(self.medium)
        self._agent_idx = AgentIndexer(field_size, self.agents)

        self._log_agent = ChannelLogger(self.agents, channels=['x', 'y'], num=self._num_alive_agents)
        # self._log_agent.log_update(self.agents)

    def _rotate_agent_buffer(self, reset_data=0.):
        tmp = self.medium
        self.medium = self.buffer_medium
        self.buffer_medium = tmp
        self.buffer_medium[:] = reset_data

    def reset(self, *,
              seed: Optional[int] = None,
              options: Optional[dict] = None
              ) -> Tuple[ObsType, dict]:
        self._init_data(self._field_size)
        return self._get_current_obs,{}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Each substep here leaves the world state valid and ready for the next substep."""

        # Motor stage
        # self._agent_move_async(action)
        self._agent_move(action)
        self._agent_deposit_and_layout(action)

        # Lifecycle stage
        energy_gain = self._agent_feed(action)
        self._agent_lifecycle()

        # Medium dynamics stage
        self._medium_resource_dynamics()
        self._medium_diffuse_decay()

        # Return rewards & statistics
        num_agents = self._num_alive_agents
        total_gain = energy_gain.sum()
        reward = float(total_gain.values)
        mean_gain = reward / num_agents if num_agents > 0 else 0.
        info = {
            'num_agents': num_agents,
            'reward': np.round(reward, 3),
            'mean_reward': np.round(mean_gain, 5)
        }
        terminated = num_agents == 0
        truncated = False

        # Then goes sensory stage by Agent
        return self._get_current_obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._renderer.render(self.medium, self.agents)

    def _medium_diffuse_decay(self):
        """Applies per-channel diffusion, channel-specific."""
        chem_ind = dict(channel='chem1')
        chem_medium = self.medium.loc[chem_ind]
        diffused = filters.gaussian(chem_medium,
                                    sigma=self.dynamics.diffuse_sigma,
                                    mode=self.dynamics.diffuse_mode,
                                    preserve_range=True,)
        diffused *= (1. - self.dynamics.rate_decay_chem)
        self.medium.loc[chem_ind] = diffused

    def _medium_resource_dynamics(self):
        """Defines agent-independent inflow & outflow of resource."""
        food_ind = dict(channel='env_food')
        self.medium.loc[food_ind] = self.dynamics.op_food_flow(self.medium.loc[food_ind])

    def _agent_move_handle_boundary(self, coords_array: da.DataArray) -> da.DataArray:
        bound = self.dynamics.boundary
        if bound == BoundaryCondition.wrap:
            return coords_array % 1.  # also handles negative overflow
        elif bound == BoundaryCondition.limit:
            return coords_array.clip(0., 1.)
        else:
            logging.warning(f'Unfamiliar boundary condition: {bound}! '
                            f'Doing nothing with boundary...')
            return coords_array

    def _agent_move(self, action: ActType):
        coord_chans = ['x', 'y']
        delta_chans = ['dx', 'dy']

        # Compute new positions, don't forget about boundary conditions
        agent_coords = self.agents.sel(channel=coord_chans).to_numpy()
        delta_coords = action.sel(channel=delta_chans).to_numpy()
        agent_coords_new = self._agent_move_handle_boundary(agent_coords + delta_coords)
        # Update agent coordinates
        self.agents.loc[dict(channel=coord_chans)] = agent_coords_new

        # Simpler logic relying on automatic coordinate alignment; untested
        # agent_coords_new = self._agent_move_handle_boundary(self.agents + action)
        # self.agents.loc[dict(channel=coord_chans)] = agent_coords_new

    def _iter_agents(self, shuffle: bool = True) -> Sequence[Tuple[int, int]]:
        xs, ys = self._get_agent_indices
        agent_inds = list(zip(xs, ys))
        if shuffle:
            np.random.default_rng().shuffle(agent_inds)  # in-place
        return agent_inds

    def _agent_move_async(self, action: ActType):
        agents = self.medium.sel(channel='agents')
        for ix, iy in self._iter_agents(shuffle=True):
            ixy = dict(x=ix, y=iy)
            # Get all medium & movement data
            agent_action = action[ixy]
            dx, dy, deposit = agent_action.values

            # Get actual agent coords (from just indices)
            x = agent_action.coords['x'].values
            y = agent_action.coords['y'].values
            # Compute new pos
            nearest_cxy = agents.sel(x=x + dx, y=y + dy, method='nearest').coords

            # Update agents
            # TODO: not working now due to bad array overwrite
            self.buffer_medium.loc[nearest_cxy] = agents[ixy]  # move agent data to new position
        self._rotate_agent_buffer()

    def _agent_deposit_and_layout(self, action: ActType):
        """Deposit chemical from agents to medium field and layout agents."""
        agent_coords = self._agent_idx.agents_to_field_coords(self.medium)
        alive_action = self._agent_idx.action_by_agents(action)
        deposit = alive_action.sel(channel='deposit1')

        # Deposit chemical
        self.medium.loc[dict(**agent_coords, channel='chem1')] += deposit.to_numpy()
        # Layout agents onto the field
        # TODO: make not binary but 'alive' continuous
        self.medium.loc[dict(channel='agents')] = 0
        self.medium.loc[dict(**agent_coords, channel='agents')] = 1.

        # # self._log_agent.log(agent_coords['x'], 'coords[x]')
        # self._log_agent.log(agent_coords['y'], 'coords[y]')

    def _agent_feed(self, action: AgtType) -> Array1C:
        """Gains food from environment and consumes internal stock"""
        # Consume food from environment
        env_stock = self.medium.sel(channel='env_food')
        consumed_field = self.dynamics.rate_feed * env_stock * self._get_agent_mask
        consumed = self._agent_idx.field_by_agents(consumed_field, only_alive=False)
        # Update food in environment
        if not self.dynamics.food_infinite:
            self.medium.loc[dict(channel='env_food')] -= consumed_field

        # Burn internal energy stock to produce action
        burned = self.dynamics.op_action_cost(action)
        gained = consumed - burned
        # Update agents array with the resulting gain
        self.agents.loc[dict(channel='agent_food')] += gained
        agent_stock = self.agents.sel(channel="agent_food")

        logging.info(f'Food consumed: {np.sum(np.array(consumed)):.3f}'
                     f', burned: {np.sum(np.array(burned)):.3f}'
                     f', total gain: {np.sum(np.array(gained)):.3f}'
                     f', agent stock: {np.sum(np.array(agent_stock)):.3f}'
                     f', env stock: {np.sum(np.array(env_stock)):.3f}'
                     )
        return gained

    def _agent_lifecycle(self):
        """Dies if consumed all internal stock,
        grows if has excess stock & favorable conditions."""
        if self.dynamics.agents_die:
            have_food = self.agents.sel(channel='agent_food') > 1e-4
            self.agents = self.agents.where(have_food, 0)

            # alive_agents = np.sum(np.array(have_food))
            # dead_agents = self._num_agents - alive_agents
            # logging.info(f'Agents alive: {alive_agents}, died: {dead_agents}')

        if self.dynamics.agents_born:
            # TODO: growing agents in favorable conditions
            # enough_neighbours
            # enough_neighbour_food
            # enough_food = self.agents.sel(channel='agent_food') > 0.5
            pass

    @property
    def _num_alive_agents(self) -> int:
        return self._agent_idx.get_alive_agents().shape[1]

    @property
    def _get_agent_mask(self) -> MaskType:
        return self.medium.sel(channel='agents') > 0

    @property
    def _get_agent_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_agent_mask.values.nonzero()

    @property
    def _get_sense_mask(self) -> MaskType:
        """Get sense mask (neighbourhood of agents)
        by diffusing agent points and rounding up."""
        agents = self.medium.sel(channel='agents')
        if self.dynamics.apply_sense_mask:
            # sigma=0.4 round=3 for square neighbourhood=1
            # sigma=0.4 round=2 for star neighbourhood=1
            # sigma=1.0 round=2 for circle neighbourhood=3
            # sigma=2.0 round=2 for circle neighbourhood=5
            sense_mask = np.ceil(filters.gaussian(agents, sigma=2.0).round(3))
        else:
            sense_mask = np.ones_like(agents)
        return sense_mask

    @property
    def _get_sensed_medium(self) -> MediumType:
        """Apply agent neighbourhood mask to the medium to get "what agents see"."""
        visible_medium = self.medium.where(self._get_sense_mask, other=0.)
        return visible_medium

    @property
    def _get_current_obs(self) -> ObsType:
        return self.agents, self._get_sensed_medium

    @property
    def _get_aspect_ratio(self) -> float:
        xs = self.medium.coords['x']
        ys = self.medium.coords['y']
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        aspect = width / height
        return aspect


if __name__ == '__main__':
    pass
