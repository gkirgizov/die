from typing import Callable, Union

import numpy as np
import xarray as da

# ActType = TypeVar('ActType')
# ObsType = TypeVar('ObsType')
ActType = da.DataArray
ObsType = da.DataArray

Array = Union[np.ndarray, da.DataArray]
Array1C = Array  # Single-channel 2D array
MaskType = Array1C
CostType = Array1C

CostOperator = Callable[[ActType], CostType]

Operator = Callable[[da.DataArray], da.DataArray]
BiOperator = Callable[[da.DataArray], da.DataArray]

medium_channels = ('agents', 'agent_food', 'env_food', 'chem1')
action_channels = ('dist', 'turn', 'deposit1')
