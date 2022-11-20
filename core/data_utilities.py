from typing import Union

import numpy as np
import torch as th


def index_select(tensor: th.Tensor, indices: th.Tensor) -> th.Tensor:
    ndims = len(tensor.shape)
    assert len(indices.shape) == 2
    assert indices.shape[1] == ndims
    i = indices.transpose(0, 1).chunk(ndims)
    return tensor[i]


def np_mask_duplicates(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Returns True mask for non-first occurences of elements"""
    mask = np.zeros(a.shape[axis], dtype=bool)
    mask[np.unique(a, return_index=True, axis=axis)[1]] = True
    return mask


def th_mask_duplicates(a: th.Tensor, axis: int = 0) -> th.Tensor:
    # TODO: test
    mask = th.zeros(a.shape[axis], dtype=bool)
    mask[th.unique(a, return_inverse=True, dim=axis)[1]] = True
    return mask


def mask_duplicates(a: Union[np.ndarray, th.Tensor], axis: int = 0) -> Union[np.ndarray, th.Tensor]:
    if isinstance(a, np.ndarray):
        return np_mask_duplicates(a, axis)
    elif isinstance(a, th.Tensor):
        return th_mask_duplicates(a, axis)
    else:
        raise TypeError()