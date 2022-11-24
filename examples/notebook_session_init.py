import numpy as np
import xarray as da
from xarray import plot
from matplotlib import pyplot as plt

from core.base_types import ActType, ObsType, MaskType, CostOperator, DataChannels
from core.data_init import DataInitializer
from core.env import Env
from core.agent import Agent
from core.utils import plot_medium


def get_test_fields(field_size):
    medium = DataInitializer(field_size, DataChannels.medium) \
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .build()
    agents = DataInitializer(field_size, DataChannels.agents) \
        .with_agents(ratio=0.05) \
        .build()
    action = DataInitializer(field_size, DataChannels.actions) \
        .with_noise('dx', 0, 3) \
        .with_noise('dy', 0, 2) \
        .with_noise('deposit1', 0, 1) \
        .build()

    return medium, agents, action


# Get & plot
medium, agents, action = get_test_fields((8, 6))
plot_medium(medium, agents)
plt.show()

# Test methods
pos = medium.sel(x=[0.13, 0.4], y=[0.15, 0.6], method='nearest')
float(action[dict(x=3,y=4)].sel(channel='deposit1').values)
action[dict(x=3,y=4)].loc[dict(channel='dx')]
