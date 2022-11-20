# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import *

import torch as th
import torch.nn as nn
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from core.data_utilities import np_mask_duplicates


def unflatten(xs):
    """ Returns 'wrapped' view on ndarray """
    return np.reshape(xs, (-1, *xs.shape))


class ObsSpec(NamedTuple):
    shape: Tuple[int, int, int]
    grid: Tuple[int, int]
    nchans: int
    ncoords: int

    @classmethod
    def from_obs(cls, obs_shape: Tuple[int, int, int]):
        return cls(obs_shape, obs_shape[:-1], obs_shape[-1], len(obs_shape)-1)

    @classmethod
    def from_grid(cls, grid_shape: Tuple[int, int], nchans: int):
        # obs_shape = (*grid_shape, nchans)
        obs_shape = (grid_shape[0], grid_shape[1], nchans)
        return cls(obs_shape, grid_shape, nchans, len(grid_shape))


class ChannelArrayWrapper:
    channels: Sequence[Tuple[str, Number, Number]] = ()

    SelectorT = Union[int, slice]

    @classmethod
    def _define_properties(cls, ids: Sequence[Tuple[str, SelectorT]]):
        for prop_name, chan_selector in ids:
            prop = property(
                fget=lambda self: self.data[:, chan_selector],
                fset=lambda self, new_data: self._bounded_assign(chan_selector, new_data)
            )
            setattr(cls, prop_name, prop)

    @classmethod
    def channel_ranges(cls):
        names, lows, highs = tuple(*zip(cls.channels))
        return lows, highs

    def __init__(self, data: np.ndarray, dtype=np.float32):
        self.data = data.astype(dtype)

    def __len__(self):
        return self.data.shape[0]

    def as_tensor(self) -> th.Tensor:
        return th.as_tensor(self.data)

    def _take_channels(self, indices):
        return self.data.take(indices, axis=-1)  # channels axis is last

    def _bounded_assign(self, selector: Union[int, slice], data: np.ndarray):
        bounds = self.channel_ranges()[selector]
        self.data[:, selector] = data.clip(*bounds)


class EnvArrayWrapper(ChannelArrayWrapper):

    channels = (
        ('NumAgents', 0., 10.),
        ('FoodAmount', 0., 100.),
        ('ChemicalConcentration1', 0., 1.),
        ('ChemicalConcentration2', 0., 1.),
    )

    def __init__(self, grid_shape: Tuple[int, int], dtype=np.float32):
        self.spec = ObsSpec.from_grid(grid_shape, len(self.channels))
        data = np.zeros(self.spec.shape, dtype=dtype)

        super().__init__(data, dtype)

        self.__class__._define_properties([
            ('num_agents', 0),
            ('food_amount', 1),
            ('chemicals', slice(2, 4)),
            ('chemicalA', 2)
        ])

    # @property
    # def num_agents(self): return self._take_channels(0)
    # @property
    # def food_amount(self): return self._take_channels(1)
    # @property
    # def chemicals(self): return self._take_channels([2, 3])
    # @property
    # def chemicalA(self): return self._take_channels(2)


class Actions:

    # TODO: possibly inlude sense
    channels = (
        ('MoveOffset', 0., 0.1),
        ('TurnRadians', -np.pi / 4, np.pi / 4),
        ('DepositTrace', 0., 1.0),
    )
    dim = len(channels)

    @classmethod
    def channel_ranges(cls):
        names, lows, highs = tuple(*zip(cls.channels))
        return lows, highs

    def __init__(self, actions: th.Tensor, mask: th.Tensor = None):
        if mask is None:
            mask = th.ones(len(actions))
        assert len(actions) == len(mask)
        self.data = actions
        self.mask = mask

    @property
    def mask_alive(self): return self.mask

    # TODO: should i return masked values?
    @property
    def offset_scale(self): return th.select(self.data, dim=-1, index=0)
    @property
    def turn_radians(self): return th.select(self.data, dim=-1, index=1)
    @property
    def deposit_trace(self): return th.select(self.data, dim=-1, index=2)


class AgentArrayWrapper(ChannelArrayWrapper):

    channels = (
        ('CoordX', 0., 1.),
        ('CoordY', 0., 1.),
        ('DirectionRadians', -np.pi / 4, np.pi / 4),
        ('SenseOffset', 0., 0.1),  # TODO: remember, sense offset is actually fixed!
        ('FoodAmount', 0., 100.),
        ('Alive', 0., 1.),
    )

    @dataclass
    class Params:
        # Food consumption per alive agent per step
        food_consumption_sustain: float = 0.2
        # Food consumption from environment
        food_consumption_env: float = 0.25

    def __init__(self, num_agents: int, params: Params = None, dtype=np.float32):
        data = np.zeros((num_agents, len(self.channels)), dtype=dtype)

        super().__init__(data, dtype=dtype)

        self.coorddim = 2 # num of coordinate
        self.params = params or AgentArrayWrapper.Params()

        self.__class__._define_properties([
            ('coords', slice(0, self.coorddim)),
            ('direction_rads', self.coorddim),
            ('sense_offsets', self.coorddim+1),
            ('food_amount', -2),
            ('alive', -1),
        ])

    # @property
    # def coords(self): return self.data[:, 0:self.coorddim]
    # @coords.setter
    # def coords(self, data): self._bounded_assign(slice(0, self.coorddim), data)
    #
    # @property
    # def direction_rads(self): return self.data[:, self.coorddim]
    # @direction_rads.setter
    # def direction_rads(self, data): self._bounded_assign(self.coorddim, data)
    #
    # @property
    # def sense_offsets(self): return self.data[:, self.coorddim+1]
    # @sense_offsets.setter
    # def sense_offsets(self, data): self._bounded_assign(self.coorddim+1, data)
    #
    # @property
    # def food_amount(self): return self.data[:, -2]
    # @food_amount.setter
    # def food_amount(self, data): self._bounded_assign(-2, data)
    #
    # @property
    # def alive(self): return self.data[:, -1]
    # @property
    # def alive_mask(self, grid_shape):
    #     return self.to_indices(grid_shape) * np.ceil(self.alive()).astype(np.int)

    def move(self, turn_radians, scale, omit_duplicates=False) -> None:
        # turn: sum turn_radians with current turn, save it
        # compute direction vector from updated angle
        # move: += offsets * directions

        assert turn_radians.shape == scale.shape

        # TODO: don't forget limiting direction through action!
        # compute & check & remove duplicates
        direction_rads = self.direction_rads + turn_radians
        destinations = self.__get_offset_coords(direction_rads, scale)

        self.direction_rads = direction_rads
        if omit_duplicates:
            duplicate_mask = np_mask_duplicates(destinations, axis=0)
            np.putmask(self.coords, not duplicate_mask, destinations)
        else:
            self.coords = destinations

    def offset_coords(self) -> np.ndarray:
        return self.__get_offset_coords(self.direction_rads, self.sense_offsets)

    def __get_offset_coords(self, radians, scale) -> np.ndarray:
        directions = np.column_stack((np.cos(radians), np.sin(radians)))
        offsets = scale * directions
        # Apply offset with cyclic overflow
        # TODO: impl if needed for other boundary conditions
        # TODO: handle case of negative result? without clip
        # result = (self.coords + offsets).clip(0.,1.)
        limited_offsets = np.fmod(self.coords + offsets, np.ones(self.coorddim)).clip(0., 2.)
        return limited_offsets

    def to_indices(self, grid_shape, with_offset: bool = True) -> np.ndarray:
        grid_scale = np.array(grid_shape) - 1
        coords = self.offset_coords() if with_offset else self.coords
        # Rescale fp-values from [0.,1.] to integer in range of grid_scale
        indices = (coords * grid_scale).round().astype(np.int)
        return indices

    def mask_dead(self) -> np.ndarray:
        return np.isclose(self.alive, 0.)


class FoodEnv(gym.Env):
    '''
    TODO: questions:
    - need init layers for: Food *In/Out*
    - Ensure this Env accepts *batched actions* i.e. actions for all agents
    '''
    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, num_agents: int = 1024, grid_shape=(128, 128), dtype=np.float32):
        self.dtype = dtype
        self.__st = EnvArrayWrapper(grid_shape)
        self.__agents = AgentArrayWrapper(num_agents, dtype)

        def fn_to_bounds(bounds):
            return np.full(grid_shape, fill_value=np.array(bounds, dtype=self.dtype))
        st_lows, st_highs = tuple(map(fn_to_bounds, self.__st.channel_ranges()))
        self.observation_space = spaces.Box( low=st_lows, high=st_highs, dtype=dtype )

        act_lows, act_highs = Actions.channel_ranges()
        self.action_space = spaces.Box( low=np.array(act_lows), high=np.array(act_highs), dtype=dtype )

    # TODO
    def init_state(self):
        pass

    def step(self, action: Actions):
        # Here I can have guarantee of unique

        # 1. Move agents
        # NB: some agents, even if they sensed different cells
        #  -- can still move to the same cell. So, omit duplicates.
        self.__agents.move(action.turn_radians, action.offset_scale, omit_duplicates=True)

        # TODO: test this in some trivial cases:
        #  2 agents, (a) one moves to other's position (b) both move to same position
        assert not np_mask_duplicates(self.__agents.coords).any()
        # The uniqueness allows to be sure that num_of_agents is always 0 or 1

        # TODO: impl setters in EnvArrayWrapper for each channel
        # 2. Deposit agents and their effect on the territory
        # 2.1 Update number of agents TODO: test
        agent_indices = self.__agents.to_indices(self.__st.spec.grid)  # NB: unique? yes! see above check.
        num_agents_env = np.zeros_like(self.__st.num_agents)
        num_agents_env[agent_indices] = 1.0  # TODO: test indexing assign
        self.__st.num_agents = num_agents_env

        # 2.2 Food consumption from environment -- create additive mask given agents map in env.
        # alive_map = num_agents.clip(0, 1).astype(np.int)
        food_env = self.__st.food_amount
        food_consumed_env = np.min(food_env, num_agents_env * self.__agents.params.food_consumption_env)
        self.__st.food_amount = food_env - food_consumed_env

        # 2.3 Food consumption for agents & their life/death cycle
        food_consumed_agents = self.__agents.params.food_consumption_env
        agent_sustain_consume = self.__agents.params.food_consumption_sustain
        food_delta_scalar = food_consumed_agents - agent_sustain_consume
        self.__agents.food_amount = (self.__agents.food_amount + action.mask_alive * food_delta_scalar).clip(0.)
        min_food_amount = 1e-3
        self.__agents.alive[self.__agents.food_amount < min_food_amount] = 0.

        # 2.4 Effects of agents on environment (chemicals)
        deposited_trace = action.deposit_trace * action.mask_alive
        self.__st.chemicals[agent_indices] += deposited_trace  # TODO: check that such assignment through this view works

        # 3. Environment dynamics pass
        # TODO: diffusion on needed channels -- through convolution / gaussian smooth in np/scipy

        # 4. Observe
        # TODO: implement offseted observe mask
        #  get sense offset indices -- and here I get mapping AgentIdx -> (iposX,iposY)
        #  "expand" that 'one-hot' map to kernel size (e.g. through hard smooth, gaussian+ceiling)
        #  apply this mask to the grid and return.

        #  BUT remember/decide: agents see circular zone? or sectors? Because making sectors isn't obvious!
        #  -> for start: they see shole circular region *at the offset distance*
        #  And the following kernels will be computed based on their *sensed* position.
        #  BUT effects will be applied to their *moved-to* position.

        # 5. Compute rewards & penalties
        # TODO: Food:
        #  rewards from newly found food,
        #  penalties from consumed food
        #  apply results to agents array -- need mapping from Agent position (2d index) to Agent array Idx, ie through select
        # TODO: penalties for actions: chemical deposition, movement distance, maybe something else

        return obs, rew, done, _
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
