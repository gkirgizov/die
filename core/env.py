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
from core.utils import plot_medium

RenderFrame = TypeVar('RenderFrame')


class BoundaryCondition(Enum):
    wrap = 'wrap'
    limit = 'limit'


@dataclass
class Dynamics:
    op_action_cost: CostOperator
    op_food_flow: FoodOperator = lambda x: x
    rate_feed: float = 0.1  # TODO: maybe do lambda taking into account input?
    rate_decay_chem: float = 0.025
    boundary: BoundaryCondition = BoundaryCondition.wrap
    diffuse_mode: str = 'wrap'
    diffuse_sigma: float = 0.5

    @staticmethod
    def default_cost(action: ActType):
        dist = np.linalg.norm(action.sel(channel=['dx', 'dy']), axis=0)
        cost = 0.5 * action.sel(channel='deposit1') + 0.25 * dist
        return cost


class Env(gym.Env[ObsType, ActType]):
    # TODO: maybe use Dataset with aligned `agents` and `medium` DataArrays
    #  with channels: x, y, food ??
    def __init__(self, field_size: Tuple[int, int]):
        self.coordgrid = self._get_meshgrid(field_size)

        self.medium = DataInitializer(field_size, DataChannels.medium) \
            .with_food_perlin(threshold=0.1) \
            .build(name='medium')

        self.agents = DataInitializer(field_size, DataChannels.agents) \
            .with_agents(ratio=0.01) \
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
        self.buffer_agents[:] = reset_data

    def _rotate_medium_buffer(self, reset_data=0.):
        tmp = self.medium
        self.medium = self.buffer_medium
        self.buffer_medium = tmp
        self.buffer_medium[:] = reset_data

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Each substep here leaves the world state valid and ready for the next substep."""

        food_got = self._agent_feed()
        self._agent_act_on_medium(action)
        food_spent = self._agent_consume_stock(action)
        self._agent_lifecycle()

        self._medium_resource_dynamics()
        self._medium_diffuse_decay()

        # NB: only agents that are alive (after lifecycle step) will move
        # self._agent_move_async(action)
        self._agent_move(action)

        num_agents = int(self._get_agent_mask.sum())
        total_fed = float(food_got.sum())
        total_spent = float(food_spent.sum())
        total_gain = max(0., total_fed - total_spent)
        mean_gain = total_gain / num_agents if num_agents > 0 else 0.
        info = {
            'num_agents': num_agents,
            'total_fed': total_fed,
            'total_spent': total_spent,
            'total_reward': total_gain,
            'mean_reward': mean_gain,
        }
        reward = total_gain
        terminated = num_agents == 0
        truncated = False

        return self._get_current_obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def plot(self, figsize=None):
        # TODO: consult aspect ratio
        plot_medium(self.medium, self.agents, figsize)

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

    def _agent_move_handle_boundary(self, coords_grid: da.DataArray) -> da.DataArray:
        bound = self.dynamics.boundary
        if bound == BoundaryCondition.wrap:
            return coords_grid % 1.  # also handles negative overflow
        elif bound == BoundaryCondition.limit:
            return coords_grid.clip(0., 1.)
        else:
            logging.warning(f'Unfamiliar boundary condition: {bound}! '
                            f'Doing nothing with boundary...')
            return coords_grid

    @property
    def _get_agent_indexer(self) -> Dict:
        ixs, iys = self._get_agent_mask.values.nonzero()
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
        pointwise_indexer = dict(x=da.DataArray(ixs),
                                 y=da.DataArray(iys))
        return pointwise_indexer

    def _agent_move(self, action: ActType):
        # Compute new positions
        delta_coords = action.sel(channel=['dx', 'dy'])
        agent_coords = self.coordgrid
        # NB: that's an array of new coords indexed by old coords -- key step
        new_coords_grid = agent_coords + delta_coords  # full grid
        # Handle boundary condition
        new_coords_grid = self._agent_move_handle_boundary(new_coords_grid)

        # Key step: elaborate approximate pointwise coordinate indexing
        agent_inds = self._get_agent_indexer  # pointwise indexer
        new_coords = new_coords_grid[agent_inds]
        # TODO: here in principle can filter out already occupied cells
        cell_indexer = dict(x=da.DataArray(new_coords[0]),
                            y=da.DataArray(new_coords[1]))
        new_cells = self.agents.sel(**cell_indexer, method='nearest')

        # Move agents with their channels
        self.buffer_agents.loc[new_cells.coords] = self.agents[agent_inds]

        self._rotate_agent_buffer()

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
            nearest_cxy = self.agents.sel(x=x + dx, y=y + dy, method='nearest').coords

            # Update agents
            # TODO: all these coords can be got in vectorized fashion? just write in buffer?
            self.buffer_agents.loc[nearest_cxy] = self.agents[ixy]  # move agent data to new position
        self._rotate_agent_buffer()

    def _agent_act_on_medium(self, action: ActType) -> Array1C:
        """Act on medium & consume required internal resource."""
        amount1 = action.sel(channel='deposit1')
        self.medium.loc[dict(channel='chem1')] += amount1
        return amount1

    def _agent_consume_stock(self, action: ActType) -> Array1C:
        consumed = self.dynamics.op_action_cost(action)
        self.agents.loc[dict(channel='agent_food')] -= consumed
        return consumed

    def _agent_feed(self) -> Array1C:
        """Gains food from environment"""
        env_stock = self.medium.sel(channel='env_food')
        gained = self.dynamics.rate_feed * env_stock
        self.agents.loc[dict(channel='agent_food')] += gained
        self.medium.loc[dict(channel='env_food')] -= gained
        return gained

    def _agent_lifecycle(self):
        """Dies if consumed all internal stock,
        grows if has excess stock & favorable conditions."""
        have_food = self.agents.sel(channel='agent_food') > 1e-4
        self.agents = self.agents.where(have_food, 0)

    @property
    def _get_agent_mask(self) -> MaskType:
        return self.agents.sel(channel='agents') > 0

    @property
    def _get_sense_mask(self) -> MaskType:
        """Get sense mask (neighbourhood of agents)
        by diffusing agent points and rounding up."""
        agents = self.agents.sel(channel='agents')
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
    def _get_agent_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_agent_mask.values.nonzero()

    @staticmethod
    def _get_meshgrid(field_size: Sequence[int]) -> np.ndarray:
        # NB: dim order is reversed in xarray
        xcs = [np.linspace(0., 1., num=size) for size in reversed(field_size)]
        coord_grid = np.stack(np.meshgrid(*xcs))
        return coord_grid


if __name__ == '__main__':
    pass
