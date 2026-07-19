"""Live matplotlib view of a running simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

from die.sim.observers import SimObserver

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from matplotlib.image import AxesImage

    from die.sim.simulation import Simulation, StepEvent

#: Backends able to show a live window, tried in order on headless-only setups.
INTERACTIVE_BACKENDS = ('macosx', 'qtagg', 'tkagg')


def _ensure_interactive_backend():
    backend = mpl.get_backend().lower()
    interactive_names = {'macosx', 'qtagg', 'qt5agg', 'qt6agg', 'tkagg',
                         'gtk3agg', 'gtk4agg', 'nbagg', 'webagg', 'wxagg'}
    if backend in interactive_names:
        return
    for candidate in INTERACTIVE_BACKENDS:
        try:
            mpl.use(candidate)
            return
        except ImportError:
            continue
    raise RuntimeError(
        f'No interactive matplotlib backend available (current: {backend!r}). '
        f'Install a GUI backend (e.g. PyQt6 or Tk) or run headless '
        f'(record to a file instead of viewing live).')


class InteractivePlotter:
    """Draws a sequence of images in a live matplotlib grid.

    The figure is created lazily on the first :meth:`draw` call,
    sized to the number of images.
    """

    def __init__(self, size: float = 6, aspect: float = 1.0, ncols: int = 2):
        self._size = size
        self._aspect = aspect
        self._ncols = ncols
        self.fig: plt.Figure | None = None
        self._artists: Sequence[AxesImage] = ()

    def _init_figure(self, images: Sequence[np.ndarray]):
        _ensure_interactive_backend()
        from matplotlib import pyplot as plt

        ncols = min(self._ncols, len(images))
        nrows = math.ceil(len(images) / ncols)
        figsize = (self._size * self._aspect * ncols, self._size * nrows)
        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                      figsize=figsize, squeeze=False)
        self.fig.tight_layout()

        artists = []
        for ax, img in zip(axes.flat, images):
            ax.tick_params(axis='both', which='both',
                           bottom=False, labelbottom=False,
                           left=False, labelleft=False)
            artists.append(ax.imshow(img))
        for ax in list(axes.flat)[len(images):]:
            ax.set_axis_off()
        self._artists = artists

        plt.ion()
        plt.show()

    def draw(self, images: Sequence[np.ndarray]):
        if self.fig is None:
            self._init_figure(images)
        else:
            for img, artist in zip(images, self._artists):
                artist.set_data(img)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

    def close(self):
        if self.fig is not None:
            from matplotlib import pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self._artists = ()


class InteractiveViewObserver(SimObserver):
    """Shows the live view of a simulation in a matplotlib window."""

    def __init__(self, every: int = 1, size: float = 6, keep_open: bool = True):
        self._every = every
        self._keep_open = keep_open
        self._plotter = InteractivePlotter(size=size)

    def on_step(self, sim: Simulation, event: StepEvent) -> None:
        if event.step % self._every == 0:
            self._plotter.draw(sim.render_frames())

    def close(self) -> None:
        # By default the window outlives the run so the final state stays
        # visible; the process owner decides when to tear it down.
        if not self._keep_open:
            self._plotter.close()
