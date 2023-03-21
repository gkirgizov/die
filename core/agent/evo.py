from typing import Tuple, Sequence, Optional

import numpy as np
import torch as th
import xarray as xa
from torch import nn

from core.agent.base import Agent
from core.base_types import ActType, ObsType, DataChannels
from core.data_init import DataInitializer
from core.utils import AgentIndexer


class EvoModel(nn.Module):
    """Model for agent learning using Evolutionary Strategies"""

    def __init__(self,
                 kernel_sizes: Sequence[int] = (3,),
                 boundary: str = 'circular',
                 p_agent_dropout: float = 0.5,
                 ):
        """
        Model: neuroevolution of Convolution kernel,
            which is a mapping of input channels to output channels
        Input: 2d env X #env_channels
        Output, model: 2d env X #action_channels
        Output, final: #agents x #action_channels (after indexing)

        If I speak about CNN (Neural CA) then I must have:
        backpropagation through agent indexing operation,
        possibly in JAX framework.

        2 ways with Evotorch:
        - Implement Env interface & use GymNE Problem class
          means properly setup obs_shape etc. data for Env
        - Use generic NEProblem class with simple eval
        """

        # TODO: agents should have access to their own resource stock

        # Determine kernel sizes:
        # - First apply kernels preserving obs shape
        # - Lastly apply the kernel mapping convolved features to actions
        num_obs_chans = len(DataChannels.medium)
        num_kernels = len(kernel_sizes)
        input_channels = [num_obs_chans] * num_kernels
        kernel_channels = [num_obs_chans] * (num_kernels-1) + [len(DataChannels.actions)]

        # TODO: do I do some sample eg from Normal distrib? or deterministic policy?
        kernels = [
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding='same',
                padding_mode=boundary,
                bias=True
            )
            for in_chans, kernel_size, out_chans
            in zip(input_channels, kernel_sizes, kernel_channels)
        ]
        # NB: notice the last activation function
        # for normalizing the outputs into [0,1] range
        kernels.append(nn.Sigmoid())
        # Dropout for breaking synchrony between agent actions
        self.agent_dropout = nn.Dropout(p=p_agent_dropout)
        # Final model
        self.kernels = nn.Sequential(*kernels)
        super().__init__()

    def forward(self, input: th.Tensor) -> th.Tensor:
        sense_transform = self.kernels(input)
        # NB: dropout can have no effect if it drops only dead cells
        output_shape = input.shape[1:]
        dropout_mask = self.agent_dropout(th.ones(output_shape))
        sense_transform *= dropout_mask
        return sense_transform


class EvoAgent(Agent):
    """Agent learning using Evolutionary Strategies"""
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 scale: float = 0.1,
                 deposit: float = 1.0,
                 ):
        self._model = model or EvoModel()
        # And these are coefficients for later scaling in forward() step
        # self.actions_coefs = np.array(Actions.channel_ranges()[1])
        self.action_coefs = th.tensor([scale, scale, deposit])
        assert len(self.action_coefs) == len(DataChannels.actions)

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        idx = AgentIndexer(medium.shape[1:], agents)
        sensed_channels = ('env_food', 'chem1')
        sense_input: xa.DataArray = medium.sel(channel=sensed_channels)

        # Call internal tensor model
        sense_input_tensor = th.as_tensor(sense_input.values)
        sense_transform_tensor = self._model(sense_input_tensor)
        sense_transform = xa.DataArray(data=sense_transform_tensor.numpy(),
                                       coords=medium.coords)

        # Get per-agent actions from sensed environment transformed by reception model
        per_agent_output = idx.field_by_agents(sense_transform, only_alive=False)
        # Rescale output actions
        per_agent_output *= self.action_coefs
        # Bulid action XArray
        action = DataInitializer.init_action_for(agents, per_agent_output)
        return action
