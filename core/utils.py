from typing import Sequence

import numpy as np
import xarray as da

from core.base_types import MediumType


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
