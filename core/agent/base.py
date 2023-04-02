import io
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Union, Any

import numpy as np

from core.base_types import ObsType, ActType, AgtType


class Agent(ABC):
    @abstractmethod
    def forward(self, obs: ObsType) -> ActType:
        """Act in the environment given observations."""
        pass

    def render(self) -> Sequence[Optional[np.ndarray]]:
        """Returns a sequence of images for agent visualization."""
        return [None]

    @property
    @abstractmethod
    def init_params(self) -> Dict[str, Any]:
        """Returns parameters from which Agent can be reconstructed."""
        pass

    def save(self, file: Union[str, os.PathLike, io.FileIO]):
        data = json.dumps(self.init_params)
        if not isinstance(file, io.FileIO):
            with open(file, 'w') as file:
                file.write(data)
        else:
            file.write(data)

    @classmethod
    def load(cls, file: Union[str, os.PathLike, io.FileIO]) -> 'Agent':
        if isinstance(file, io.FileIO):
            params_data: Dict = json.load(file)
        else:
            with open(file, 'r') as f:
                params_data: Dict = json.load(f)
        return cls(**params_data)

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
