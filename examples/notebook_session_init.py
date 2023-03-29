import numpy as np
import xarray as da
from matplotlib import pyplot as plt

from core.base_types import DataChannels
from core.data_init import DataInitializer
from core.plotting import InteractivePlotter


def get_test_fields(field_size, agents_ratio=0.2):
    medium = DataInitializer(field_size, DataChannels.medium) \
        .with_agents(ratio=agents_ratio) \
        .with_food_perlin(threshold=0.5) \
        .with_chem(threshold=0.25) \
        .build()
    agents = DataInitializer.agents_from_medium(medium)
    s = 0.5
    action = DataInitializer.action_for(agents) \
        .with_noise('dx', -s, s) \
        .with_noise('dy', -s, s) \
        .with_noise('deposit1', 0, 1) \
        .build_agents()
    return medium, agents, action


# Get & plot
field_sz = (8, 6)
medium, agents, action = get_test_fields(field_sz)

# Alive mask indexing
alive = agents.sel(channel='alive')
action.where(alive > 0.).dropna(dim='index')

# Receipt for data assignment by approximate coords
pos_range = medium.sel(x=[0.13, 0.4],
                       y=[0.15, 0.6],
                       method='nearest')
# Pointwise
# https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
pos_pointwise = medium.sel(x=da.DataArray([0.13, 0.4], dims='z'),
                           y=da.DataArray([0.15, 0.6], dims='z'),
                           method='nearest')
pos = pos_pointwise
new_data = np.random.random(pos.shape).round(2)
medium.loc[pos.coords] = new_data

# Getting values
float(action[dict(x=3,y=4)].sel(channel='deposit1').values)
# Chained indexing for assignment
action[dict(x=3,y=4)].loc[dict(channel='dx')]
