from typing import Tuple, Sequence, Callable

import numpy as np
from matplotlib import cm as cm

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
        return cm.get_cmap(cmap)(monochrome_data)

    @staticmethod
    def _set_colors(field_colors_id) -> Callable:
        if field_colors_id == 'random':
            color = (np.random.random(3) - 0.5) * 2
        else:
            color = RendererBase.field_colors.get(field_colors_id)
        if color is not None:
            color /= np.linalg.norm(color)
            rgb_mapper=lambda rgb: np.cross(color, rgb, axisb=-1)
        else:
            rgb_mapper=lambda rgb: rgb
        return rgb_mapper

    def __init__(self, field_size: Tuple[int, int]):
        self.field_size = field_size

    @property
    def img_shape(self) -> Tuple[int, int]:
        width, height = self.field_size
        img_shape = (height, width)
        return img_shape

    def _upd_img_dummy(self, *args) -> np.ndarray:
        width, height = self.field_size
        return np.ones((height, width, 4))


class EnvRenderer(RendererBase):
    def __init__(self,
                 field_size: Tuple[int, int],
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
        ones_channel = np.ones(self.img_shape)
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
