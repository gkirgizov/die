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


class ChannelNormalizer:
    """Rescales unbounded scalar fields to [0, 1] for display.

    Keeps one scale per named channel, so channels that live on wildly
    different scales stay individually readable -- chem1 climbs past 4 over a
    run while env_food never leaves 0.5, and a single shared scale would sink
    the latter into the black.

    Two details make it usable on live simulation data:

    - **Percentile, not max.** The fields are heavy-tailed: a handful of hot
      pixels sit several times above the 99th percentile, and scaling by the
      max would squash everything else back into the dark.
    - **Smoothed over time.** The scale is an EMA, so a channel drifting
      across a run stays visible without the whole pane pulsing frame to frame.

    Note this trades away cross-channel comparison: after normalizing, equal
    brightness in two channels no longer means equal concentration.
    """

    def __init__(self,
                 percentile: float = 99.,
                 decay: float = 0.9,
                 min_scale: float = 0.05):
        self._percentile = percentile
        self._decay = decay
        self._min_scale = min_scale
        self._scales: dict[str, float] = {}

    def __call__(self, field: np.ndarray, key: str) -> np.ndarray:
        values = np.asarray(field, dtype=float)
        level = float(np.percentile(values, self._percentile))
        previous = self._scales.get(key)
        # First sighting seeds the EMA outright, else the channel would spend
        # its first frames scaled towards an arbitrary starting point.
        scale = level if previous is None \
            else self._decay * previous + (1. - self._decay) * level
        self._scales[key] = scale
        # Floor the divisor: a near-empty field (a run whose agents died out)
        # has a tiny percentile, and dividing by it amplifies dust into noise.
        return values / max(scale, self._min_scale)


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
    #: Medium channels shown as R, G, B -- all unbounded concentrations.
    medium_channels = ('agents', 'env_food', 'chem1')

    def __init__(self,
                 field_size: tuple[int, int],
                 is_trace_colored: bool = True,
                 field_colors_id: str = 'rgb',
                 normalize: bool = True):
        self._is_trace_colored = is_trace_colored
        self._rgb_mapper = self._set_colors(field_colors_id)
        self._agent_trace = FieldTrace(field_size)
        self._normalizer = ChannelNormalizer() if normalize else None
        super().__init__(field_size)

    def _scaled(self, field, key: str) -> np.ndarray:
        """Rescales an unbounded concentration channel, if normalizing.

        Only for channels that are quantities. Masks (the alive/alpha channel)
        and signed fields (an agent's gradient render) carry meaning that
        per-channel rescaling would destroy, so they bypass this.
        """
        if self._normalizer is None:
            return np.asarray(field, dtype=float)
        return self._normalizer(field, key)

    def render(self, medium: MediumType, agents: AgtType) -> ImagesType:
        frames = [self._upd_img_medium(medium),
                  self._upd_img_trace(medium),
                  self._upd_img_agents(agents),
                  ]
        return frames

    def _upd_img_medium(self, medium: MediumType) -> np.ndarray:
        """Returns RGB/RGBA image ready for imshow"""
        # transpose dimensions so that channels dim would be the last
        medium_data_rgb = np.stack(
            [self._scaled(medium.sel(channel=channel), channel)
             for channel in self.medium_channels], axis=-1)
        medium_data_rgb = self._rgb_mapper(medium_data_rgb)
        return medium_data_rgb

    def _upd_img_trace(self, medium: MediumType) -> np.ndarray:
        self._agent_trace.update(medium.sel(channel='agents'))
        # Scale before colormapping: the colormap maps [0, 1] and clips the
        # rest, so an unbounded trace would arrive already flattened.
        trace_channel = self._scaled(self._agent_trace.as_mask(), 'trace')

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
        # NB: the alive mask becomes the alpha channel -- it is a mask, not a
        # quantity, so it must stay binary and is never rescaled.
        alive_mask = agents_data_rgb[:, :, 0].astype(bool)
        agent_food = self._scaled(agents_data_rgb[:, :, 1], 'agent_food')
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

    def __init__(self, env: Env, agent: Agent, field_colors_id: str = 'rgb',
                 normalize: bool = True):
        self._env = env
        self._agent = agent
        field_size = tuple(env.medium.shape[1:])
        self._renderer = EnvRenderer(field_size, field_colors_id=field_colors_id,
                                     normalize=normalize)

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
