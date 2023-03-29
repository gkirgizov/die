from itertools import chain
from typing import Tuple, Sequence

from matplotlib import pyplot as plt
from matplotlib.image import AxesImage

from core.agent.base import Agent
from core.env import Env
from core.render import ImagesType, RendererCallable


class InteractivePlotter:
    @staticmethod
    def get(env: Env, agent: Agent):
        return InteractivePlotter(env.render, agent.render)

    def __init__(self,
                 *renderers: RendererCallable,
                 size: float = 6,
                 aspect: float = 1.0,
                 ion: bool = True,
                 ):
        # Prepare renderers
        self._renderers = renderers
        # Prepare initial images
        images = self._render_images()
        # Prepare matching number of axes with needed cols/rows
        self.fig, self._axes = InteractivePlotter._init_axes(images, size, aspect)
        # Prepare artists through ordered correspondence
        self._artists = InteractivePlotter._init_artists(self._axes, images)

        # self._drawers = [
        #     ([medium_ax, trace_ax, agent_ax], self._renderers[0]),
        #     ([spare_ax], self._renderers[1]),
        # ]
        if ion:
            plt.ion()

    @staticmethod
    def _init_axes(images: Sequence,
                   size: float, aspect: float
                   ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        # TODO: Calculate number of subplots from number of images
        # Setup figure with subplots
        figheight = size
        figwidth = size * aspect
        # 4-grid
        figsize = (figwidth * 2, figheight * 2)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                gridspec_kw={'width_ratios': [1, 1],
                                             'height_ratios': [1, 1]})
        (medium_ax, trace_ax), (agent_ax, spare_ax) = axs
        # Single-row
        # figsize = (figwidth * 2, figheight)
        # self.fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        # (medium_ax, trace_ax) = axs

        fig.tight_layout()
        InteractivePlotter._disable_ticks(medium_ax)
        InteractivePlotter._disable_ticks(trace_ax)
        InteractivePlotter._disable_ticks(agent_ax)
        InteractivePlotter._disable_ticks(spare_ax)

        all_axes = [medium_ax, trace_ax, agent_ax, spare_ax]
        return fig, all_axes

    @staticmethod
    def _init_artists(axes: Sequence[plt.Axes], images: Sequence) -> Sequence[AxesImage]:
        artists = []
        for ax, img in zip(axes, images):
            artist = None if img is None else ax.imshow(img)
            artists.append(artist)
        return artists

    @staticmethod
    def _disable_ticks(ax: plt.Axes):
        # Set display of agent array with grid lines etc.
        ax.tick_params(axis='both', which='both',
                       bottom=False, labelbottom=False,
                       left=False, labelleft=False,
                       )

    def _render_images(self) -> ImagesType:
        all_images = list(chain(*(render() for render in self._renderers)))
        return all_images

    def update(self):
        if not self._artists:
            raise ValueError("First call initial `show` for initialising images.")

        for img, artist in zip(self._render_images(), self._artists):
            if img is None:
                continue
            artist.set_data(img)

    def draw(self):
        self.update()
        # TODO: add consideration of 'is_visible'
        plt.draw()
        plt.pause(0.01)
