from typing import Tuple, Sequence

import numpy as np
import torch as th
from torch import nn

from core.agent.base import Agent
from core.base_types import ActType, ObsType, DataChannels
from core.data_init import DataInitializer
from core.utils import AgentIndexer


class EvoAgent(Agent):
    """Agent learning using Evolutionary Strategies"""

    def __init__(self,
                 kernel_sizes: Sequence[int] = (3,),
                 boundary: str = 'circular',
                 p_agent_dropout: float = 0.5,
                 scale: float = 0.1,
                 deposit: float = 1.0):
        """
        Model: neuroevolution of Convolution kernel,
            which is a mapping of input channels to output channels
        Input: 2d env X #env_channels
        Output, model: 2d env X #action_channels
        Output, final: #agents x #action_channels (after indexing)

        If I speak about CNN (Neural CA) then I must have:
        backpropagation through agent indexing operation,
        possibly in JAX framework.
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

        # And these are coefficients for later scaling in forward() step
        # self.actions_coefs = np.array(Actions.channel_ranges()[1])
        self.action_coefs = np.array([scale, scale, deposit])
        assert len(self.action_coefs) == len(DataChannels.actions)

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        output_shape = medium.shape[1:]
        idx = AgentIndexer(medium.shape[1:], agents)

        sensed_channels = ('env_food', 'chem1')
        sense_input = medium.sel(channel=sensed_channels)
        sense_transform = self.kernels(sense_input)

        per_agent_output = idx.field_by_agents(sense_transform, only_alive=False)
        # NB: dropout can have no effect if it drops only dead cells
        dropout_mask = self.agent_dropout(th.ones(output_shape))
        per_agent_output *= dropout_mask
        # Rescale output actions
        per_agent_output *= self.action_coefs

        action = DataInitializer.init_action_for(agents, per_agent_output)
        return action
