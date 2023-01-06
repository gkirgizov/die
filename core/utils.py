from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as da
from xarray import plot

from core.base_types import MediumType, AgtType


def plot_medium(medium: MediumType, agents: AgtType,
                size: int = 8, aspect: float = 1.0,
                with_grid_agents=False):
    # Setup figure with subplots
    figheight = size
    figwidth = size * aspect
    figsize = (figwidth * 2, figheight)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                            gridspec_kw={'width_ratios': [1, 1]})
    medium_ax, agent_ax = axs
    cmap = 'viridis'

    # Setup medium plot
    medium_data = medium.sel(channel=['agents', 'env_food', 'chem1'])
    # transpose dimensions so that channes dim would be the last
    medium_data_rgb = medium_data.values.transpose((1, 2, 0))

    # Setup plot of agents
    width, height = medium_data_rgb.shape[:2]
    img_shape = (height, width)
    shape = (2, height, -1)
    agent_channels = ['alive', 'agent_food']
    agents_data_rgb = agents.sel(channel=agent_channels) \
        .values.reshape(shape).transpose((1, 2, 0))
    # Construct RGBA channels
    # Make dead cells appear as white space for contrast
    alive_mask = agents_data_rgb[:, :, 0].astype(bool)
    agent_food = agents_data_rgb[:, :, 1]
    zero_channel = np.zeros(img_shape)
    agents_data_rgb = np.stack([zero_channel,
                                agent_food,
                                zero_channel,
                                alive_mask,
                                ], axis=-1)

    # Set display of agent array with grid lines etc.
    agent_ax.tick_params(axis='both', which='both',
                         bottom=False, labelbottom=False,
                         left=False, labelleft=False,
                         )
    if with_grid_agents:
        agent_ax.set_xticks(np.arange(0, width))
        agent_ax.set_yticks(np.arange(0, height))
        agent_ax.xaxis.grid(True)
        agent_ax.yaxis.grid(True)

    medium_ax.imshow(medium_data_rgb)
    agent_ax.imshow(agents_data_rgb)


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
