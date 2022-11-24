from typing import Callable, Union, Sequence, Hashable

import numpy as np
import xarray as da

# ActType = TypeVar('ActType')
# ObsType = TypeVar('ObsType')
ActType = da.DataArray
AgtType = da.DataArray
ObsType = da.DataArray

Array = Union[np.ndarray, da.DataArray]
Array1C = Array  # Single-channel 2D array
MaskType = Array1C
CostType = Array1C

CostOperator = Callable[[ActType], CostType]

Operator = Callable[[da.DataArray], da.DataArray]
BiOperator = Callable[[da.DataArray], da.DataArray]

Channels = Sequence[Hashable]


class DataChannels:
    medium: Channels = ('env_food', 'chem1')
    agents: Channels = ('agents', 'agent_food')
    # action: Channels = ('dist', 'turn', 'deposit1')
    actions: Channels = ('dx', 'dy', 'deposit1')
