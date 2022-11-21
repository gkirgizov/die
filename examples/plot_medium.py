import numpy as np
import xarray as da
from matplotlib import pyplot as plt
from xarray import plot

from core.base_types import ActType, ObsType, MaskType, CostOperator
from core.data_init import DataInitializer
from core.env import Env
from core.utils import plot_medium


def try_plot(field_size=(256, 256)):
    medium_data = DataInitializer(field_size)\
        .with_agents(ratio=0.05) \
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .build()

    medium = Env.init_medium_array(field_size, init_data=medium_data)

    plot_medium(medium)
    plt.show()


if __name__ == '__main__':
    try_plot((512,512))
