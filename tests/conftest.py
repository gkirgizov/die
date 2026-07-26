import numpy as np
import pytest
import torch as th

from die.env import Dynamics, Env


@pytest.fixture
def small_field() -> tuple[int, int]:
    return (16, 16)


@pytest.fixture
def seeded():
    np.random.seed(42)
    th.manual_seed(42)


@pytest.fixture
def small_env(small_field, seeded) -> Env:
    return Env(small_field, Dynamics(init_agent_ratio=0.2))
