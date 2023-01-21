from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as da
from xarray import plot

from core.base_types import MediumType, AgtType


class FieldTrace:
    def __init__(self,
                 field_size: Tuple[int, int],
                 trace_steps: int = 8,
                 ):
        self._decay = 1 - 1 / trace_steps
        self._trace_field = np.zeros(field_size)

    @property
    def trace(self) -> np.ndarray:
        return self._trace_field

    def as_mask(self, inverse=False):
        if inverse:
            return 1. - self._trace_field
        else:
            return self._trace_field

    def update(self, field):
        self._trace_field = self._trace_field * self._decay + field


class EnvDrawer:
    def __init__(self,
                 field_size: Tuple[int, int],
                 size: float = 8,
                 aspect: float = 1.0,
                 with_grid_agents=False,
                 with_agent_trace=False):
        # Setup figure with subplots
        figheight = size
        figwidth = size * aspect
        figsize = (figwidth * 2, figheight * 2)
        self.fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [1, 1],
                                                  'height_ratios': [1, 1]})

        self.field_size = field_size
        (medium_ax, trace_ax), (agent_ax, spare_ax) = axs
        self._agent_trace = FieldTrace(field_size)
        self._disable_ticks(trace_ax)
        self._disable_ticks(agent_ax, with_grid=with_grid_agents)
        self._disable_ticks(spare_ax)

        self._drawers = [
            (medium_ax, self._upd_img_medium),
            (agent_ax, self._upd_img_agents),
            (trace_ax, self._upd_img_trace),
            (spare_ax, self._upd_img_dummy),
        ]
        self._artists = []
        # self.cmap = 'viridis'
        plt.ion()

    def _disable_ticks(self, ax: plt.Axes, with_grid: bool = False):
        width, height = self.field_size
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

    @property
    def img_shape(self) -> Tuple[int, int]:
        width, height = self.field_size
        img_shape = (height, width)
        return img_shape

    def _upd_img_dummy(self, *args) -> np.ndarray:
        width, height = self.field_size
        return np.ones((height, width, 3))

    def _upd_img_medium(self, medium: MediumType, agents: AgtType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # Setup medium plot
        medium_data = medium.sel(channel=['agents', 'env_food', 'chem1'])
        # transpose dimensions so that channes dim would be the last
        medium_data_rgb = medium_data.values.transpose((1, 2, 0))
        return medium_data_rgb

    def _upd_img_trace(self, medium: MediumType, agents: AgtType) -> np.ndarray:
        self._agent_trace.update(medium.sel(channel='agents'))
        trace_channel = self._agent_trace.as_mask()
        zero_channel = np.zeros(self.img_shape)
        data_rgb = np.stack([trace_channel,
                             trace_channel,
                             trace_channel,
                             ], axis=-1)
        return data_rgb

    def _upd_img_agents(self, medium: MediumType, agents: AgtType) -> np.ndarray:
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
        for ax, drawer in self._drawers:
            img = drawer(medium, agents)
            artist = ax.imshow(img)
            self._artists.append(artist)

    def draw(self, medium: MediumType, agents: AgtType):
        if not self._artists:
            raise ValueError("First call initial `show` for initialising images.")

        for (ax, drawer), artist in zip(self._drawers, self._artists):
            img = drawer(medium, agents)
            artist.set_data(img)

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


def polar2z(r, theta):
    return r * np.exp(np.multiply(1j, theta))


def z2polar(z):
    return abs(z), np.angle(z)


def polar2xy(r, theta):
    z = polar2z(r, theta)
    return np.real(z), np.imag(z)


def xy2polar(x, y):
    return z2polar(x + np.multiply(1j, y))
