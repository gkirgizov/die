from typing import Callable, Union

import numpy as np
import xarray as da

# ActType = TypeVar('ActType')
# ObsType = TypeVar('ObsType')
ActType = da.DataArray
ObsType = da.DataArray

MaskType = Union[np.ndarray, da.DataArray]  # Single-channel medium mask
CostType = Union[np.ndarray, da.DataArray]  # Single-channel DataArray

CostOperator = Callable[[ActType], CostType]

Operator = Callable[[da.DataArray], da.DataArray]
BiOperator = Callable[[da.DataArray], da.DataArray]
