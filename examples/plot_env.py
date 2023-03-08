import time

from matplotlib import pyplot as plt

from core.base_types import DataChannels
from core.data_init import DataInitializer
from core.plotting import EnvDrawer


def try_plot(field_size=(256, 256)):
    medium = DataInitializer(field_size, DataChannels.medium) \
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .with_agents(ratio=0.15) \
        .build(name='medium')
    agents = DataInitializer.agents_from_medium(medium)

    drawer = EnvDrawer(field_size)
    drawer.show(medium, agents)
    drawer.draw(medium, agents)
    plt.show()
    time.sleep(5)


if __name__ == '__main__':
    try_plot((256, 256))
