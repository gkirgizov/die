import numpy as np
import xarray as da
from matplotlib import pyplot as plt
from xarray import plot

from core.base_types import ActType, ObsType, MaskType, CostOperator, DataChannels
from core.data_init import DataInitializer
from core.env import Env
from core.utils import plot_medium


def try_plot(field_size=(256, 256)):
    medium_data = DataInitializer(field_size, DataChannels.medium)\
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .build()
    agents_data = DataInitializer(field_size, DataChannels.agents) \
        .with_agents(ratio=0.05) \
        .build()

    medium = Env.init_field_array(field_size, channels=DataChannels.medium, init_data=medium_data)
    agents = Env.init_field_array(field_size, channels=DataChannels.agents, init_data=agents_data)

    plot_medium(medium, agents)
    plt.show()


if __name__ == '__main__':
    try_plot((256, 256))
