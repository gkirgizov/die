import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np
import xarray as da
from skimage import filters

from die import utils
from die.data_init import DataInitializer
from die.types import (
    ActType,
    AgtType,
    Array1C,
    CostOperator,
    DataChannels,
    FoodOperator,
    MaskType,
    MediumType,
    ObsType,
)
from die.utils import AgentIndexer, ChannelLogger


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
    # defines if actions with cost higher than available resource are truncated
    strict_cost: bool = True
    food_infinite: bool = False
    agents_die: bool = False
    agents_born: bool = False

    # Initialization options
    init_agent_ratio: float = 0.1
    #: Abundance of the initial food field: a plain multiplier on the perlin
    #: patches. The scarce-to-rich axis of an environment family.
    food_scale: float = 1.0
    #: Perlin frequency of the food field -- few octaves give large smooth
    #: basins, many give fine speckle. The patchiness axis.
    food_octaves: int = 8
    #: Upper mask on the raw perlin field; see
    #: :meth:`die.data_init.DataInitializer.with_food_perlin`.
    food_threshold: float = 1.0


class Env(gym.Env[ObsType, ActType]):
    def __init__(self,
                 field_size: tuple[int, int],
                 dynamics: Dynamics | None = None):
        self._field_size = field_size
        self.coordgrid = utils.get_meshgrid(field_size)
        self.dynamics = dynamics or Dynamics()
        # Deep copies of the initial (medium, agents) arrays keyed by seed.
        # Random field init (esp. perlin noise) is expensive; repeated seeded
        # resets (e.g. per-candidate evaluations during training) restore
        # the snapshot instead of regenerating it.
        self._init_cache: dict[int, tuple[MediumType, AgtType]] = {}
        self._init_data(field_size)

    def _init_data(self, field_size: tuple[int, int]):
        self.medium = DataInitializer(field_size, DataChannels.medium) \
            .with_const('env_food', 0.5) \
            .with_food_perlin(threshold=self.dynamics.food_threshold,
                              octaves=self.dynamics.food_octaves,
                              scale=self.dynamics.food_scale) \
            .with_agents(ratio=self.dynamics.init_agent_ratio) \
            .build(name='medium')
        self.buffer_medium = self.medium.copy(deep=True)

        self.agents = DataInitializer.agents_from_medium(self.medium)
        self._agent_idx = AgentIndexer(field_size, self.agents)

        self._log_agent = ChannelLogger(self.agents, channels=['x', 'y'],
                                        num=self._num_alive_agents)
        # self._log_agent.log_update(self.agents)

    def _rotate_agent_buffer(self, reset_data=0.):
        tmp = self.medium
        self.medium = self.buffer_medium
        self.buffer_medium = tmp
        self.buffer_medium[:] = reset_data

    def reset(self, *,
              seed: int | None = None,
              options: dict | None = None
              ) -> tuple[ObsType, dict]:
        super().reset(seed=seed)
        if seed is not None:
            # Field init draws from the global numpy RNG and (perlin noise)
            # from the stdlib `random` module -- seed both.
            random.seed(seed)
            np.random.seed(seed)
        if seed is not None and seed in self._init_cache:
            self._restore_init_snapshot(seed)
        else:
            self._init_data(self._field_size)
            if seed is not None:
                self._store_init_snapshot(seed)
        return self.observe(), {}

    def observe(self) -> ObsType:
        """Returns the current observation without stepping the env.

        NOTE: the returned arrays alias live env state -- do not mutate them.
        """
        return self._get_current_obs

    def _store_init_snapshot(self, seed: int):
        self._init_cache[seed] = (self.medium.copy(deep=True),
                                  self.agents.copy(deep=True))

    def _restore_init_snapshot(self, seed: int):
        medium, agents = self._init_cache[seed]
        self.medium = medium.copy(deep=True)
        self.buffer_medium = self.medium.copy(deep=True)
        self.agents = agents.copy(deep=True)
        self._agent_idx = AgentIndexer(self._field_size, self.agents)
        self._log_agent = ChannelLogger(self.agents, channels=['x', 'y'],
                                        num=self._num_alive_agents)

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
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

    def _iter_agents(self, shuffle: bool = True) -> Sequence[tuple[int, int]]:
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
        # Only alive agents extract & burn energy: dead agent slots are parked
        # at (0, 0) and would otherwise phantom-consume from that cell.
        alive_mask = self.agents.sel(channel='alive').to_numpy() > 0.
        gained = (consumed - burned) * alive_mask
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
            # NB: in-place update -- rebinding self.agents would desync
            # the AgentIndexer, which holds a reference to this array
            self.agents.values[:] = self.agents.where(have_food, 0).values

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
    def _get_agent_indices(self) -> tuple[np.ndarray, np.ndarray]:
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
