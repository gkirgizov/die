from typing import Tuple

import numpy as np
from perlin_noise import PerlinNoise

from core.base_types import medium_channels


class DataInitializer:
    def __init__(self, field_size: Tuple[int, int]):
        self._size = field_size
        self._channels = {chan: np.zeros(field_size)
                          for chan in medium_channels}

    def _mask(self, sampled: np.ndarray, mask_above_threshold: float = 1.0) -> np.ndarray:
        mask = sampled < mask_above_threshold
        return sampled * mask

    def _get_random(self, a=0., b=1.) -> np.ndarray:
        return (b-a) * np.random.random_sample(self._size) + a

    def _get_perlin(self, octaves: int = 8) -> np.ndarray:
        noise = PerlinNoise(octaves=octaves)
        xs = np.linspace(0, 1, self._size[0])
        ys = np.linspace(0, 1, self._size[1])
        field = np.array([[noise((x, y)) for y in ys] for x in xs])
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

    def with_agents(self, ratio: float):
        agents = np.ceil(self._mask(self._get_random(), mask_above_threshold=ratio))
        self._channels['agents'] = agents
        self._channels['agent_food'] = agents * self._get_random(0.1, 0.2)
        return self

    def with_food_perlin(self, threshold: float = 0.25):
        self._channels['env_food'] = self._mask(self._get_perlin(), threshold)
        return self

    def with_chem(self, threshold: float = 0.1):
        self._channels['chem1'] = self._mask(self._get_perlin(), threshold)
        return self

    def build(self) -> np.ndarray:
        return np.stack(list(self._channels.values()))
