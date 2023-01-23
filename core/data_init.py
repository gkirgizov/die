from itertools import cycle
from numbers import Number
from numbers import Number
from typing import Tuple, Optional, List, Iterable
from typing import Union, Sequence, Hashable

import numpy as np
import xarray as da
from perlin_noise import PerlinNoise

from core import utils
from core.base_types import Channels, DataChannels, ObsType, ActType, MediumType, AgtType, FoodOperator
from core.utils import get_agents_by_medium


class FieldSequence(Sequence[np.ndarray]):
    def __init__(self,
                 field_size: Sequence[int],
                 dt: float = 0.01,
                 t_bounds: Tuple[float, float] = (0, 10),
                 ):
        self._size = field_size
        self._tbounds = t_bounds
        self._grid = utils.get_meshgrid(field_size)
        self._xs = np.linspace(0, 1, self._size[0])
        self._ys = np.linspace(0, 1, self._size[1])
        self._ts = np.arange(*t_bounds, dt)

    def get_flow_operator(self,
                          scale: float = 1.0,
                          decay: float = 0.0,
                          ) -> FoodOperator:
        it = iter(self)

        def food_flow(current):
            return scale * next(it) + (1 - decay) * current

        return food_flow

    def __iter__(self):
        for t in cycle(self._ts):
            yield self[t]

    def __len__(self):
        return len(self._ts)

    def __contains__(self, t: float):
        t0, t_end = self._tbounds
        return t0 <= t < t_end

    def __getitem__(self, t: float) -> np.ndarray:
        raise NotImplementedError


class PerlinNoiseSequence(FieldSequence):
    def __init__(self,
                 field_size: Sequence[int],
                 dt: float = 0.01,
                 t_bounds: Tuple[float, float] = (0, 1),
                 octaves: int = 8):
        super().__init__(field_size, dt, t_bounds)
        self._noise = PerlinNoise(octaves=octaves)

    def __getitem__(self, t: float) -> np.ndarray:
        return np.array([[self._noise((x, y, t))
                          for y in self._ys]
                         for x in self._xs]
                        ).round(3)


class WaveSequence(FieldSequence):
    def __getitem__(self, t: float) -> np.ndarray:
        # Waves running
        pi = np.pi
        x, y = (self._grid - 0.5) * 2  # normalized to [-1, 1]
        r = np.linalg.norm((x, y), axis=0)
        rwave = r + np.cos(pi * x) + np.sin(0.4 * pi * y)
        z_waves = np.cos(1 * pi * (rwave + t))

        # Islands moving
        sx, sy = 3, 3
        z_islands = (np.sin(pi * x * sx + t) + np.cos(pi * y * sy + t))

        # Make a mixture of both
        mix = 0.25
        z = (1 - mix) * z_waves + mix * z_islands  # v1
        # z = z_waves * z_islands  # v2

        return z


class DataInitializer:

    @staticmethod
    def init_field_array(field_size: Tuple[int, int],
                         channels: Sequence[Hashable],
                         name: Optional[str] = None,
                         init_data: Optional[Union[Number, np.ndarray]] = None) -> MediumType:
        shape = (len(channels), *field_size)
        xs = np.linspace(0, 1, field_size[0])
        ys = np.linspace(0, 1, field_size[1])

        if init_data is None:
            init_data = 0

        medium = da.DataArray(
            data=init_data,
            dims=('channel', 'x', 'y'),
            coords={'x': xs, 'y': ys, 'channel': list(channels)},
            name=name,
        )
        return medium

    @staticmethod
    def init_agential_array(num_agents: int,
                            channels: Sequence[Hashable],
                            name: Optional[str] = None,
                            init_data: Optional[Union[Number, np.ndarray]] = None) -> da.DataArray:
        """Creates flat int-indexed array"""
        index = np.arange(0, num_agents)
        if init_data is None:
            init_data = 0

        channel_array = da.DataArray(
            data=init_data,
            dims=['channel', 'index'],
            coords={'channel': list(channels), 'index': index},
            name=name,
        )
        return channel_array

    @staticmethod
    def agents_from_medium(medium: MediumType, max_agents=None, food_ratio=1.0) -> AgtType:
        channels: Sequence[Hashable] = DataChannels.agents
        name: str = 'agents'

        agents_coords = get_agents_by_medium(medium)
        num_agents_alive = agents_coords.shape[-1]
        chan_alive = np.ones(num_agents_alive)
        chan_food = DataInitializer.get_random(num_agents_alive, 0.1, food_ratio)
        init_data = np.vstack([agents_coords, chan_alive, chan_food])

        if not max_agents:
            max_agents = medium.shape[-2] * medium.shape[-1]
        shape = (len(channels), max_agents)
        all_init_data = np.zeros(shape)
        all_init_data[:, :num_agents_alive] = init_data

        agents = DataInitializer.init_agential_array(max_agents, channels, name, all_init_data)
        return agents

    @staticmethod
    def init_action_for(agents: AgtType, init_data=0.) -> ActType:
        return DataInitializer.init_agential_array(num_agents=agents.shape[-1],
                                                   channels=DataChannels.actions,
                                                   name='actions',
                                                   init_data=init_data)

    @staticmethod
    def action_for(agents: AgtType) -> 'DataInitializer':
        return DataInitializer(field_size=agents.shape[-1],
                               channels=DataChannels.actions,
                               name='actions')

    @staticmethod
    def get_random(size, a=0., b=1.) -> np.ndarray:
        return (b-a) * np.random.random_sample(size).round(3) + a

    def __init__(self,
                 field_size: Tuple[int, int],
                 channels: Optional[Channels] = None,
                 name: Optional[str] = None):
        self._size = field_size
        self._channels = {chan: np.zeros(field_size) for chan in channels or ()}
        self._name = name

    def _mask(self, sampled: np.ndarray,
              mask_below: float = 0.0,
              mask_above: float = 1.0) -> np.ndarray:
        mask = (mask_below <= sampled) & (sampled <= mask_above)
        return sampled * mask

    def _get_random(self, a=0., b=1.) -> np.ndarray:
        return self.get_random(self._size, a, b)

    def _get_perlin(self, octaves: int = 8) -> np.ndarray:
        # TODO: abstract for any dim
        noise = PerlinNoise(octaves=octaves)
        xs = np.linspace(0, 1, self._size[0])
        ys = np.linspace(0, 1, self._size[1])
        field = np.array([[noise((x, y)) for y in ys] for x in xs]).round(3)
        return field

    # TODO: get figue
    def _get_circle(self, center=(0.5, 0.5), radius=0.5):
        return np.zeros(self._size)

    def _get_corners(self, size=0.1):
        total = 0
        for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            circle = self._get_circle(corner, size)
            total += circle
        return total

    def _add_masked(self, channel: str, data: np.ndarray):
        mask = self._channels[channel] > 0.
        self._channels[channel] += data * mask
        return self

    def with_const(self, channel: str, value=0.):
        self._channels[channel] = np.full(self._size, value, dtype=np.float)
        return self

    def with_noise(self, channel: str, a=0, b=1):
        self._channels[channel] = self._get_random(a, b)
        return self

    def with_agents(self, ratio: float):
        agents = np.ceil(self._mask(self._get_random(), mask_above=ratio))
        self._channels['agents'] = agents
        # self._channels['agent_food'] = agents * self._get_random(0.1, 0.2)
        return self

    def with_food_perlin(self, threshold: float = 0.25, octaves: int = 8):
        self._channels['env_food'] = self._mask(self._get_perlin(octaves=octaves),
                                                mask_above=threshold)
        return self

    def with_chem(self, threshold: float = 0.1):
        self._channels['chem1'] = self._mask(self._get_perlin(octaves=24),
                                             mask_above=threshold)
        return self

    def build_numpy(self) -> np.ndarray:
        return np.stack(list(self._channels.values()))

    def build(self, name: Optional[str] = None) -> MediumType:
        data = self.build_numpy()
        return DataInitializer.init_field_array(field_size=self._size,
                                                channels=self._channels,
                                                name=name,
                                                init_data=data)
