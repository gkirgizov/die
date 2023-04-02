import io
import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Union

import numpy as np

from core.base_types import ObsType, ActType, AgtType


class Agent(ABC):
    @abstractmethod
    def forward(self, obs: ObsType) -> ActType:
        pass

    def render(self) -> Sequence[Optional[np.ndarray]]:
        return [None]

    @abstractmethod
    def save(self, file: Union[str, os.PathLike, io.FileIO]):
        pass

    @classmethod
    @abstractmethod
    def load(cls, file: Union[str, os.PathLike, io.FileIO]):
        pass

    @staticmethod
    def postprocess_action(agents: AgtType, action: ActType) -> ActType:
        masked = Agent._masked_alive(agents, action)
        rescaled = Agent._rescale_outputs(masked)
        return rescaled

    # TODO: rescaling: seems unnecessary because of [0, 1] bounds?
    @staticmethod
    def _rescale_outputs(action: ActType) -> ActType:
        """Scale outputs to their natural boundaries."""
        return action

    # TODO: agent masking: seems unnecessary because of natural Env logic?
    @staticmethod
    def _masked_alive(agents: AgtType, action: ActType) -> ActType:
        agent_alive_mask = agents.sel(channel='alive') > 0
        masked_action = action * agent_alive_mask
        return masked_action
