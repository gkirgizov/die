from numbers import Number
from numbers import Number
from typing import Tuple, Optional, List
from typing import Union, Sequence, Hashable

import numpy as np
import xarray as da
from perlin_noise import PerlinNoise

from core.base_types import Channels, DataChannels, ObsType, ActType, MediumType, AgtType
from core.utils import get_agents_by_medium


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

        agents = da.DataArray(
            data=all_init_data,
            dims=['channel', 'index'],
            coords={'channel': list(channels)},
            name=name,
        )
        return agents

    @staticmethod
    def init_action_for(medium: MediumType, init_data=0.) -> ActType:
        return DataInitializer.init_field_array(field_size=medium.shape[1:],
                                                channels=DataChannels.actions,
                                                name='actions',
                                                init_data=init_data)

    @staticmethod
    def action_for(medium: MediumType) -> 'DataInitializer':
        return DataInitializer(field_size=medium.shape[1:],
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

    def with_food_perlin(self, threshold: float = 0.25):
        self._channels['env_food'] = self._mask(self._get_perlin(),
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
