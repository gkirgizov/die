import inspect
import logging
import warnings
from copy import deepcopy
from typing import Sequence, Dict, Callable, Any

import matplotlib
import numpy as np
import xarray
import xarray as da
import torch as th
import torch.nn as nn
from evotorch import Solution
from evotorch.neuroevolution import NEProblem

from core.base_types import MediumType, AgtType, FieldIdx, ActType


class AgentIndexer:

    def __init__(self, field_size, agents: AgtType):
        self.__agents = agents  # read-only here
        # Map of floating point coords to numpy array indices
        self.__imesh = get_indices_mesh(field_size)

    def agents_to_field_coords(self, field: da.DataArray, only_alive=True) -> FieldIdx:
        coord_chans = ['x', 'y']
        # Agents array stores real-valued coordinates,
        #  so need first to get cells and only then get their coordinates.
        agent_cells = self.field_by_agents(field, only_alive)
        agent_coords = {ch: agent_cells.coords[ch] for ch in coord_chans}
        return agent_coords

    def action_by_agents(self, action: ActType) -> da.DataArray:
        alive = self.__agents.sel(channel='alive')
        masked_action = action.where(alive > 0.).dropna(dim='index')
        return masked_action

    def field_by_agents(self, field: da.DataArray, only_alive=False, offset=0.) -> da.DataArray:
        """Returns array of channels selected from field per agent in agents array.

        :param field:
        :param only_alive:
        :param offset: global or per agent offset for indexing the field.
        """
        coord_chans = ['x', 'y']
        # Pointwise approximate indexing:
        #  get mapping AGENT_IDX->ACTION as a sequence
        #  details: https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
        agents = self.get_alive_agents() if only_alive else self.__agents
        idx_coords = agents.sel(channel=coord_chans) + offset
        cell_indexer = dict(zip(coord_chans, idx_coords))
        field_selection = field.sel(**cell_indexer, method='nearest')
        return field_selection

    def tensor_by_agents(self, tensor: th.Tensor, only_alive=False, offset=0.) -> da.DataArray:
        # Get numeric integer coordinates given index-mesh
        indices = self.field_by_agents(self.__imesh, only_alive, offset)
        # Select tensor elements given integer coordinates
        # TODO: somehow index all channels at once, or work with flattened indices
        idx = indices.values
        ixs = idx[0]
        iys = idx[1]
        selection = tensor[:, ixs, iys]
        return selection

    def get_alive_agents(self, view=False) -> AgtType:
        """Returns only agents which are alive, dropping dead agents from array."""
        agent_inds = (self.__agents.sel(channel='alive') > 0).values.nonzero()[0]
        if view:
            alive_agents = self.__agents.isel(index=agent_inds)
        else:
            indexer = dict(index=da.DataArray(agent_inds))
            alive_agents = self.__agents[indexer]
        return alive_agents


class ChannelLogger:
    def __init__(self,
                 init_array: da.DataArray,
                 channels: Sequence[str],
                 num: int = -1):
        self.num = num
        self.chs = channels
        self.data = 0.
        self.delta = 0.
        self.update(init_array[:, :self.num])
        self._logger = print
        # self._logger = logging.debug

    def log_update(self, array: da.DataArray):
        self.update(array[:, :self.num])
        self.log(self.delta, 'delta')
        self.log(self.data, 'data ')

    def update(self, array: da.DataArray):
        new_data = array.sel(channel=self.chs).to_numpy()
        self.delta = new_data - self.data
        self.data = new_data

    def log(self, data, prefix=None, prec=3):
        with np.printoptions(threshold=np.inf):
            prefix = f'{prefix}: ' if prefix else ''
            d = np.asarray(data).round(prec)
            self._logger(f'{prefix}{np_info(d)}')
            self._logger(f'{d}')

    def log_nonzero(self, field):
        non_zero = np.count_nonzero(field.sel(channel='agents'))
        self._logger(f'num_nonzer0={non_zero}')


def get_meshgrid(field_size: Sequence[int]) -> np.ndarray:
    # NB: dim order is reversed in xarray
    xcs = [np.linspace(0., 1., num=size) for size in reversed(field_size)]
    coord_grid = np.stack(np.meshgrid(*xcs))
    # steps = [abs(coords[1] - coords[0]) for coords in xcs]
    return coord_grid


def get_indices_mesh(field_size: Sequence[int],
                     coordnames: Sequence[str] = tuple('xyzuvw')) -> xarray.DataArray:
    """Returns DataArray with float coords which maps them to their array indices.
    Allows approximate mapping of coords (XArray's strength) into numpy array indices."""
    ndims = len(field_size)
    # NB: dim order is reversed in xarray
    coordnames = list(reversed(coordnames[:ndims]))
    # NB: don't confuse str-indices of the coords with dim names
    coords = {'coord': coordnames}
    mesh_coords = [np.linspace(0., 1., num=size) for size in reversed(field_size)]
    mesh_coords = dict(zip(coordnames, mesh_coords))
    coords.update(mesh_coords)

    indices = [np.arange(size) for size in field_size]
    indices_mesh = np.stack(np.meshgrid(*indices))
    indices_map = xarray.DataArray(data=indices_mesh, coords=coords)
    return indices_map


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


def get_radians(coords):
    x, y = coords
    _, rads = xy2polar(x, y)
    return rads


def renormalize_radians(rads):
    """Renormalizes radians in (-np.pi, np.pi] interval."""
    return (rads - np.pi) % (-2 * np.pi) + np.pi


def discretize(value, step):
    return (value // step) * step


def print_angles(prefix, radians):
    print(prefix, np.rad2deg(radians).astype(int))


def np_info(grad):
    return (f'shape: {np.shape(grad)}, unique: {len(np.unique(grad))}, '
            f'max: {np.max(grad).round(3)}, min: {np.min(grad).round(3)}, '
            f'avg: {np.mean(grad).round(3)}, std: {np.std(grad).round(3)}')


def setup_logging(level=logging.INFO, disable_warnings: bool = False):
    # setup logging
    logging.basicConfig(level=level)
    # disable matplotlib warnings, mabye put that into InteractivePlotter
    matplotlib.pyplot.set_loglevel('error')
    # disable extra warnings not controlled by logging
    if disable_warnings or level > logging.WARNING:
        warnings.filterwarnings('ignore')


def make_net(problem: NEProblem, solution: Solution) -> nn.Module:
    """No-copying version of ``evotorch.NEProblem.make_net()``."""
    parameters = solution.access_values(keep_evals=True)
    with th.no_grad():
        net = problem.parameterize_net(parameters)
    return net


def save_args(fun: Callable, locs: Dict[str, Any]) -> Dict[str, Any]:
    """Saves function arguments with values in a dict, including
    all default parameters. Requires `locals()` to be passed."""
    sig = inspect.signature(fun)
    args = {name: deepcopy(locs.get(name, param.default))
            for name, param in sig.parameters.items()}
    return args
