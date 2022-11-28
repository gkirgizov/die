from typing import Optional, Union, List, Tuple, TypeVar, Sequence

import numpy as np
import xarray as da
from xarray import plot

from core.base_types import ActType, ObsType, MaskType, CostOperator, MediumType, AgtType


def plot_medium(medium: MediumType, agents: AgtType, **imshow_kwargs):
    agents_data = agents.sel(channel='agents')
    medium_data = medium.sel(channel=['env_food', 'chem1'])
    rgb_data_to_plot = da.concat([agents_data, medium_data], dim='channel')

    cmap = 'viridis'
    artist = plot.imshow(rgb_data_to_plot,
                         rgb=medium.dims[0],
                         x=medium.dims[1],
                         y=medium.dims[2],
                         add_labels=True,
                         cmap=cmap,
                         **imshow_kwargs
                         )
    return artist


def get_meshgrid(field_size: Sequence[int]) -> np.ndarray:
    # NB: dim order is reversed in xarray
    xcs = [np.linspace(0., 1., num=size) for size in reversed(field_size)]
    coord_grid = np.stack(np.meshgrid(*xcs))
    return coord_grid
