from typing import Optional, Union, List, Tuple, TypeVar, Callable

import numpy as np
import xarray as da
import gymnasium as gym

from core.base_types import ActType, ObsType, MaskType


class Agent:
    def __init__(self):
        self.model: Callable[[ObsType], ActType] = None

    def forward(self, obs: ObsType) -> ActType:
        action = self.model(obs)
        scaled = self._rescale_outputs(action)
        return scaled

    def _rescale_outputs(self, action: ActType) -> ActType:
        """Scale outputs to their natural boundaries."""
        pass
