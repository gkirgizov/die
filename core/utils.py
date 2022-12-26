from typing import Optional, Union, List, Tuple, TypeVar, Sequence

import numpy as np
import xarray as da
from xarray import plot

from core.base_types import ActType, ObsType, MaskType, CostOperator, MediumType, AgtType


def plot_medium(medium: MediumType, agents: AgtType, **imshow_kwargs):
    medium_data = medium.sel(channel=['agents', 'env_food', 'chem1'])
    rgb_data_to_plot = medium_data

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
    # steps = [abs(coords[1] - coords[0]) for coords in xcs]
    return coord_grid


def get_agents_by_medium(medium: MediumType) -> np.ndarray:
    coord_chans = ['x', 'y']
    agent_mask = medium.sel(channel='agents') > 0
    axes_inds = agent_mask.values.nonzero()
    agent_inds_pointwise = {coord: da.DataArray(inds)
                            for coord, inds in zip(coord_chans, axes_inds)}

    # now can get coords from selected medium
    sel_medium = medium[agent_inds_pointwise]
    coord_chans = np.array([np.array(sel_medium[c]) for c in coord_chans])

    return coord_chans
