from typing import Sequence, Dict

import numpy as np
import xarray as da

from core.base_types import MediumType, AgtType


class AgentIndexer:

    def __init__(self, agents: AgtType):
        self.__agents = agents  # read-only here

    def agents_to_field_coords(self, field: da.DataArray) -> Dict:
        coord_chans = ['x', 'y']
        # Agents array stores real-valued coordinates,
        #  so need first to get cells and only then get their coordinates.
        agent_cells = self.field_by_agents(field, only_alive=False)
        agent_coords = {ch: agent_cells.coords[ch] for ch in coord_chans}
        return agent_coords

    def field_by_agents(self, field: da.DataArray, only_alive=True) -> da.DataArray:
        """Returns array of channels selected from field per agent in agents array."""
        coord_chans = ['x', 'y']
        # Pointwise approximate indexing:
        #  get mapping AGENT_IDX->ACTION as a sequence
        #  details: https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
        agents = self.get_alive_agents() if only_alive else self.__agents
        cell_indexer = {coord_ch: agents.sel(channel=coord_ch)
                        for coord_ch in coord_chans}
        field_selection = field.sel(**cell_indexer, method='nearest')
        return field_selection

    def get_alive_agents(self, view=False) -> AgtType:
        """Returns only agents which are alive, dropping dead agents from array."""
        agent_inds = (self.__agents.sel(channel='alive') > 0).values.nonzero()[0]
        if view:
            alive_agents = self.__agents.isel(index=agent_inds)
        else:
            indexer = dict(index=da.DataArray(agent_inds))
            alive_agents = self.__agents[indexer]
        return alive_agents


def get_meshgrid(field_size: Sequence[int]) -> np.ndarray:
    # NB: dim order is reversed in xarray
    xcs = [np.linspace(0., 1., num=size) for size in reversed(field_size)]
    coord_grid = np.stack(np.meshgrid(*xcs))
    # steps = [abs(coords[1] - coords[0]) for coords in xcs]
    return coord_grid


def get_agents_by_medium(medium: MediumType) -> np.ndarray:
    coord_chans = ['x', 'y']
    agent_mask = medium.sel(channel='agents') > 0
    axes_inds = agent_mask.values.nonzero()
    agent_inds_pointwise = {coord: da.DataArray(inds)
                            for coord, inds in zip(coord_chans, axes_inds)}

    # now can get coords from selected medium
    sel_medium = medium[agent_inds_pointwise]
    coord_chans = np.array([np.array(sel_medium[c]) for c in coord_chans])

    return coord_chans


def polar2z(r, theta):
    return r * np.exp(np.multiply(1j, theta))


def z2polar(z):
    return abs(z), np.angle(z)


def polar2xy(r, theta):
    z = polar2z(r, theta)
    return np.real(z), np.imag(z)


def xy2polar(x, y):
    return z2polar(x + np.multiply(1j, y))


def renormalize_radians(rads):
    """Renormalizes radians in (-np.pi, np.pi] interval."""
    return (rads - np.pi) % (-2 * np.pi) + np.pi


def discretize(value, step):
    return (value // step) * step
