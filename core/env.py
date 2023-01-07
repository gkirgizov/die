import logging
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Optional, Union, List, Tuple, TypeVar, Sequence, Hashable, Dict

import numpy as np
# from scipy.signal import fftconvolve as convolve
import skimage
from skimage import filters
import xarray as da
import gymnasium as gym

from core.base_types import DataChannels, ActType, ObsType, MaskType, CostOperator, AgtType, MediumType, FoodOperator, \
    Array1C
from core.data_init import DataInitializer
from core import utils

RenderFrame = TypeVar('RenderFrame')


class BoundaryCondition(Enum):
    wrap = 'wrap'
    limit = 'limit'


def linear_action_cost(action: ActType, weights=(0.02, 0.01)) -> Array1C:
    # Default cost of actions is computed as linear combination of:
    #  total crossed distance plus chemical deposition
    dist = np.linalg.norm(action.sel(channel=['dx', 'dy']), axis=0)
    deposit = action.sel(channel='deposit1')
    cost = weights[0] * deposit + weights[1] * dist
    return cost


def zero_cost(action: ActType) -> Array1C:
    return np.zeros(action.shape[1:])


@dataclass
class Dynamics:
    op_action_cost: CostOperator = linear_action_cost
    op_food_flow: FoodOperator = lambda x: x
    rate_feed: float = 0.1  # TODO: maybe do lambda taking into account input?
    rate_decay_chem: float = 0.025
    boundary: BoundaryCondition = BoundaryCondition.wrap
    diffuse_mode: str = 'wrap'
    diffuse_sigma: float = 0.5

    # test options?
    food_infinite: bool = False
    agents_die: bool = False
    agents_born: bool = False

    # Initialization options
    init_agent_ratio: float = 0.1


class Env(gym.Env[ObsType, ActType]):
    # TODO: maybe use Dataset with aligned `agents` and `medium` DataArrays
    #  with channels: x, y, food ??
    def __init__(self,
                 field_size: Tuple[int, int],
                 dynamics: Optional[Dynamics] = None):
        self.coordgrid = utils.get_meshgrid(field_size)
        # self.coordsteps = np.abs(self.coordgrid[:, 1] - self.coordgrid[:, 0])

        self.dynamics = dynamics or Dynamics(rate_feed=0.1,
                                             rate_decay_chem=0.001)

        self.medium = DataInitializer(field_size, DataChannels.medium) \
            .with_food_perlin(threshold=1.0) \
            .with_agents(ratio=self.dynamics.init_agent_ratio) \
            .build(name='medium')

        self.agents = DataInitializer.agents_from_medium(self.medium)

        self.buffer_medium = self.medium.copy(deep=True)

    def _rotate_agent_buffer(self, reset_data=0.):
        tmp = self.medium
        self.medium = self.buffer_medium
        self.buffer_medium = tmp
        self.buffer_medium[:] = reset_data

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Each substep here leaves the world state valid and ready for the next substep."""

        # NB: only agents that are alive (after lifecycle step) will move
        # self._agent_move_async(action)
        self._agent_move(action)
        self._agents_to_medium()  # updates position of agents in medium array

        energy_gain = self._agent_feed(action)
        self._agent_act_on_medium(action)
        self._agent_lifecycle()

        self._medium_resource_dynamics()
        self._medium_diffuse_decay()

        num_agents = self._num_agents
        total_gain = energy_gain.sum()
        reward = float(total_gain.values)
        mean_gain = reward / num_agents if num_agents > 0 else 0.
        info = {
            'num_agents': num_agents,
            'reward': reward,
            'mean_reward': mean_gain,
        }
        terminated = num_agents == 0
        truncated = False

        return self._get_current_obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def plot(self, size: float = 8):
        return utils.plot_medium(self.medium, self.agents,
                                 size=size, aspect=self._get_aspect_ratio)

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

    def _sel_by_agents(self, field: da.DataArray, only_alive=False) -> da.DataArray:
        """Returns array of channels selected from field per agent in agents array."""
        coord_chans = ['x', 'y']
        # Pointwise approximate indexing:
        #  get mapping AGENT_IDX->ACTION as a sequence
        #  details: https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
        # TODO: alive selection doesn't work
        agents = self._get_alive_agents() if only_alive else self.agents
        cell_indexer = {coord_ch: agents.sel(channel=coord_ch)
                        for coord_ch in coord_chans}
        field_selection = field.sel(**cell_indexer, method='nearest')
        return field_selection

    def _medium_agent_coords(self) -> Dict:
        # coord_chans = self.medium.coords[1:]
        coord_chans = ['x', 'y']
        agent_cells = self._sel_by_agents(self.medium)
        agent_coords = {ch: agent_cells.coords[ch] for ch in coord_chans}
        return agent_coords

    def _agent_move(self, action: ActType):
        coord_chans = ['x', 'y']
        delta_chans = ['dx', 'dy']

        # Map action *grid* to *1d* agent-like array
        agent_actions = self._sel_by_agents(action)
        # Compute new positions, don't forget about boundary conditions
        agent_coords = self.agents.sel(channel=coord_chans).to_numpy()
        delta_coords = agent_actions.sel(channel=delta_chans).to_numpy()
        agent_coords_new = self._agent_move_handle_boundary(agent_coords + delta_coords)
        # Update agent coordinates
        self.agents.loc[dict(channel=coord_chans)] = agent_coords_new

        # Simpler logic relying on automatic coordinate alignment; untested
        # agent_coords_new = self._agent_move_handle_boundary(self.agents + agent_actions)
        # self.agents.loc[dict(channel=coord_chans)] = agent_coords_new

        # TODO: maybe deposit at this same step? other actions, no?

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

    def _agents_to_medium(self):
        agent_coords = self._medium_agent_coords()
        # Set agent locations
        # TODO: make not binary but 'alive' continuous
        self.medium.loc[dict(channel='agents')] = 0
        self.medium.loc[dict(**agent_coords, channel='agents')] = 1

    def _agent_act_on_medium(self, action: ActType) -> Array1C:
        """Act on medium & consume required internal resource."""
        # Deposit chemical
        amount1 = action.sel(channel='deposit1')
        self.medium.loc[dict(channel='chem1')] += amount1 * self._get_agent_mask
        return amount1

    def _agent_feed(self, action: AgtType) -> Array1C:
        """Gains food from environment and consumes internal stock"""
        # Consume food from environment
        env_stock = self.medium.sel(channel='env_food')
        consumed = self.dynamics.rate_feed * env_stock * self._get_agent_mask
        # Update food in environment
        if not self.dynamics.food_infinite:
            self.medium.loc[dict(channel='env_food')] -= consumed

        # Burn internal energy stock to produce action
        burned = self.dynamics.op_action_cost(action)
        gained = consumed - burned
        # Update agents array with the resulting gain
        per_agent_gain = self._sel_by_agents(gained, only_alive=False)
        self.agents.loc[dict(channel='agent_food')] += per_agent_gain
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


    def _get_alive_agents(self, view=True) -> AgtType:
        """Returns only agents which are alive, dropping dead agents from array."""
        agent_inds = self.agents.sel(channel='alive').values.nonzero()[0]
        alive_agents = self.agents.isel(index=agent_inds) if view else self.agents[agent_inds]
        return alive_agents

    @property
    def _num_agents(self) -> int:
        return self._get_alive_agents().shape[1]

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
        # sigma=0.4 round=3 for square neighbourhood=1
        # sigma=0.4 round=2 for star neighbourhood=1
        sense_mask = np.ceil(filters.gaussian(agents, sigma=0.4).round(3))
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
