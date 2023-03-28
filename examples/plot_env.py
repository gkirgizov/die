import time

import numpy as np

from core.base_types import DataChannels
from core.data_init import DataInitializer
from core.plotting import InteractivePlotter
from core.render import EnvRenderer


def try_plot(field_size=(256, 256)):
    medium = DataInitializer(field_size, DataChannels.medium) \
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .with_agents(ratio=0.15) \
        .build(name='medium')
    agents = DataInitializer.agents_from_medium(medium)

    renderer = EnvRenderer(field_size, field_colors_id='rgb')

    def env_render():
        return renderer.render(medium, agents)

    def agt_render():
        return np.zeros((*field_size, 3))

    drawer = InteractivePlotter(env_render, agt_render, size=6)

    # Initial show
    drawer.draw()
    time.sleep(3)

    # Update image and show again
    medium *= 0.5
    agents.loc[dict(channel='agent_food')] *= 2
    drawer.draw()
    time.sleep(3)


if __name__ == '__main__':
    try_plot((256, 256))
