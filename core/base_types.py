from typing import Callable, Union, Sequence, Hashable, Tuple

import numpy as np
import xarray as da

# ActType = TypeVar('ActType')
# ObsType = TypeVar('ObsType')
ActType = da.DataArray
AgtType = da.DataArray  # 1-channel array
MediumType = da.DataArray
ObsType = Tuple[AgtType, MediumType]

Array = Union[np.ndarray, da.DataArray]
Array1C = Array  # Single-channel 2D array
MaskType = Array1C
CostType = Array1C

CostOperator = Callable[[ActType], CostType]
FoodOperator = Callable[[MediumType], Array1C]

Operator = Callable[[da.DataArray], da.DataArray]
BiOperator = Callable[[da.DataArray], da.DataArray]

Channels = Sequence[Hashable]


class DataChannels:
    medium: Channels = ('agents', 'env_food', 'chem1')
    agents: Channels = ('x', 'y', 'alive', 'agent_food')
    # agents: Channels = ('agents', 'agent_food')
    # action: Channels = ('dist', 'turn', 'deposit1')
    actions: Channels = ('dx', 'dy', 'deposit1')

    # TODO: align
    #  dx|x
    #  agents | alive
    #  env_food | agent_food
