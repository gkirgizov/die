from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

from die.types import AgtType, MediumType

if TYPE_CHECKING:
    from die.agents.base import Agent
    from die.env import Env


class FieldTrace:
    def __init__(self,
                 field_size: tuple[int, int],
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


ImagesType = Sequence[np.ndarray]
RendererCallable = Callable[[], ImagesType]


class RendererBase:
    """Base class for helpers of rendering entities"""

    field_colors = {
        'rgb': None,
        'one': [0.19, -0.3, 0.74],
        'two': [-0.45, 0.65, 0.83],
        'random': None,
    }

    @staticmethod
    def _colorify(monochrome_data: np.array, cmap: str ='gray'):
        return matplotlib.colormaps[cmap](monochrome_data)

    @staticmethod
    def _set_colors(field_colors_id) -> Callable:
        if field_colors_id == 'random':
            color = (np.random.random(3) - 0.5) * 2
        else:
            color = RendererBase.field_colors.get(field_colors_id)
        if color is not None:
            color /= np.linalg.norm(color)

            def rgb_mapper(rgb):
                return np.cross(color, rgb, axisb=-1)
        else:
            def rgb_mapper(rgb):
                return rgb
        return rgb_mapper

    def __init__(self, field_size: tuple[int, int]):
        self.field_size = field_size

    @property
    def img_shape(self) -> tuple[int, int]:
        width, height = self.field_size
        img_shape = (height, width)
        return img_shape

    def _upd_img_dummy(self, *args) -> np.ndarray:
        width, height = self.field_size
        return np.ones((height, width, 4))


class EnvRenderer(RendererBase):
    def __init__(self,
                 field_size: tuple[int, int],
                 is_trace_colored: bool = True,
                 field_colors_id: str = 'rgb'):
        self._is_trace_colored = is_trace_colored
        self._rgb_mapper = self._set_colors(field_colors_id)
        self._agent_trace = FieldTrace(field_size)
        super().__init__(field_size)

    def render(self, medium: MediumType, agents: AgtType) -> ImagesType:
        frames = [self._upd_img_medium(medium),
                  self._upd_img_trace(medium),
                  self._upd_img_agents(agents),
                  ]
        return frames

    def _upd_img_medium(self, medium: MediumType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # transpose dimensions so that channels dim would be the last
        medium_data_rgb = np.stack([medium.sel(channel='agents'),
                                    medium.sel(channel='env_food'),
                                    medium.sel(channel='chem1'),
                                    ], axis=-1)
        medium_data_rgb = self._rgb_mapper(medium_data_rgb)
        return medium_data_rgb

    def _upd_img_trace(self, medium: MediumType) -> np.ndarray:
        self._agent_trace.update(medium.sel(channel='agents'))
        trace_channel = self._agent_trace.as_mask()

        cmap_id = 'magma' if self._is_trace_colored else 'gray'
        trace_colored = self._colorify(trace_channel, cmap_id)

        return trace_colored

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


def to_display_range(frame: np.ndarray) -> np.ndarray:
    """Clips a float image to the [0, 1] range image consumers expect.

    Rendered channels are unbounded (medium concentrations and agent food
    grow past 1), while ``imshow`` accepts only [0, 1] floats -- it clips
    anything else itself and logs a warning per frame, which floods the
    terminal and breaks the progress bar. Clipping here keeps the live view
    and the recorded animation (which clips in :func:`to_uint8_rgb`) identical.
    """
    return np.clip(frame, 0., 1.)


class FrameSource:
    """Produces the visual frames for an (env, agent) pair.

    Owns the (stateful) :class:`EnvRenderer` -- e.g. the decaying agent
    trace -- so one instance should serve one simulation run at a time.
    """

    def __init__(self, env: Env, agent: Agent, field_colors_id: str = 'rgb'):
        self._env = env
        self._agent = agent
        field_size = tuple(env.medium.shape[1:])
        self._renderer = EnvRenderer(field_size, field_colors_id=field_colors_id)

    def frames(self) -> list[np.ndarray]:
        env_frames = self._renderer.render(self._env.medium, self._env.agents)
        agent_frames = self._agent.render()
        return [to_display_range(frame)
                for frame in [*env_frames, *agent_frames] if frame is not None]

    __call__ = frames


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Converts a float [0, 1] RGB/RGBA (or grayscale) image to uint8 RGB.

    RGBA images are composited onto a black background.
    """
    img = np.asarray(frame, dtype=float)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        rgb, alpha = img[..., :3], img[..., 3:4]
        img = rgb * alpha
    return (np.clip(img, 0., 1.) * 255).astype(np.uint8)


def _resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[:2] == (height, width):
        return img
    row_idx = (np.arange(height) * img.shape[0] / height).astype(int)
    col_idx = (np.arange(width) * img.shape[1] / width).astype(int)
    return img[row_idx][:, col_idx]


def compose_grid(frames: Sequence[np.ndarray], ncols: int = 2) -> np.ndarray:
    """Tiles frames into a single uint8 RGB image grid.

    Frames of differing sizes (e.g. an agent's 1x1 fallback frame) are
    nearest-neighbor resized to the largest frame size.
    """
    if not frames:
        raise ValueError('No frames to compose')
    tiles = [to_uint8_rgb(frame) for frame in frames]
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    tiles = [_resize_nearest(tile, height, width) for tile in tiles]

    ncols = min(ncols, len(tiles))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    while len(tiles) % ncols != 0:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + ncols]) for i in range(0, len(tiles), ncols)]
    return np.vstack(rows)
