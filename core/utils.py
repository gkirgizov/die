from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as da
from xarray import plot

from core.base_types import MediumType, AgtType


class EnvDrawer:

    def __init__(self,
                 field_size: Tuple[int, int],
                 size: float = 8,
                 aspect: float = 1.0,
                 with_grid_agents=False):
        # Setup figure with subplots
        figheight = size
        figwidth = size * aspect
        figsize = (figwidth * 2, figheight)
        self.fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [1, 1]})

        self.field_size = field_size
        self.medium_ax, self.agent_ax = axs
        self._init_agent_ax(with_grid=with_grid_agents)
        self._artist_medium = None
        self._artist_agents = None
        # self.cmap = 'viridis'
        plt.ion()

    def _init_agent_ax(self, with_grid: bool):
        width, height = self.field_size
        ax = self.agent_ax
        # Set display of agent array with grid lines etc.
        ax.tick_params(axis='both', which='both',
                             bottom=False, labelbottom=False,
                             left=False, labelleft=False,
                             )
        if with_grid:
            ax.set_xticks(np.arange(0, width))
            ax.set_yticks(np.arange(0, height))
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)

    def _upd_img_medium(self, medium: MediumType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # Setup medium plot
        medium_data = medium.sel(channel=['agents', 'env_food', 'chem1'])
        # transpose dimensions so that channes dim would be the last
        medium_data_rgb = medium_data.values.transpose((1, 2, 0))
        return medium_data_rgb

    def _upd_img_agents(self, agents: AgtType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # Setup plot of agents
        width, height = self.field_size
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
        return agents_data_rgb

    def show(self, medium: MediumType, agents: AgtType):
        """Called for initial show at interactive usage or for static show"""
        medium_data_rgb = self._upd_img_medium(medium)
        self._artist_medium = self.medium_ax.imshow(medium_data_rgb)

        agents_data_rgb = self._upd_img_agents(agents)
        self._artist_agents = self.agent_ax.imshow(agents_data_rgb)

    def draw(self, medium: MediumType, agents: AgtType):
        if not self._artist_medium or not self._artist_agents:
            raise ValueError("First call initial `show` for initialising images.")

        medium_data_rgb = self._upd_img_medium(medium)
        self._artist_medium.set_data(medium_data_rgb)

        agents_data_rgb = self._upd_img_agents(agents)
        self._artist_agents.set_data(agents_data_rgb)

        # TODO: add consideration of 'is_visible'
        plt.draw()
        plt.pause(0.01)


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
