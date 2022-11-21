from typing import Optional, Union, List, Tuple, TypeVar

import numpy as np
import xarray as da
from xarray import plot

from core.base_types import ActType, ObsType, MaskType, CostOperator


def plot_medium(medium: ObsType, figsize=None):
    rgb_data_to_plot = medium.sel(channel=['agents', 'env_food', 'chem1'])
    cmap = 'viridis'
    artist = plot.imshow(rgb_data_to_plot,
                         rgb=medium.dims[0],
                         x=medium.dims[1],
                         y=medium.dims[2],
                         figsize=figsize,
                         add_labels=True,
                         cmap=cmap,
                         )
    return artist
