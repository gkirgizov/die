from typing import Tuple, Optional, Callable

import numpy as np
from matplotlib import pyplot as plt

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
                 color_mapper: Optional[Callable] = None):
        # Setup figure with subplots
        figheight = size
        figwidth = size * aspect
        figsize = (figwidth * 2, figheight * 2)
        self.fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [1, 1],
                                                  'height_ratios': [1, 1]})
        self._rgb_mapper = color_mapper or (lambda rgba: rgba)

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
        return np.ones((height, width, 4))

    def _upd_img_medium(self, medium: MediumType, agents: AgtType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # transpose dimensions so that channels dim would be the last
        ones_channel = np.ones(self.img_shape)
        medium_data_rgb = np.stack([medium.sel(channel='agents'),
                                    medium.sel(channel='env_food'),
                                    medium.sel(channel='chem1'),
                                    ], axis=-1)
        medium_data_rgb = self._rgb_mapper(medium_data_rgb)
        return medium_data_rgb

    def _upd_img_trace(self, medium: MediumType, agents: AgtType) -> np.ndarray:
        self._agent_trace.update(medium.sel(channel='agents'))
        trace_channel = self._agent_trace.as_mask()
        ones_channel = np.ones(self.img_shape)
        data_rgb = np.stack([trace_channel,
                             trace_channel,
                             trace_channel,
                             ones_channel,
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
