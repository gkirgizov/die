from typing import Tuple, Sequence, Optional

import numpy as np
import torch as th
import xarray as xa
from torch import nn
from torch.nn.init import xavier_uniform

from core.agent.base import Agent
from core.base_types import ActType, ObsType, DataChannels, MediumType
from core.data_init import DataInitializer
from core.utils import AgentIndexer


class ConvolutionModel(nn.Module):
    """Model for agent perception based on convolution kernels."""

    def __init__(self,
                 num_obs_channels: int = 3,
                 num_act_channels: int = 3,
                 kernel_sizes: Sequence[int] = (3,),
                 boundary: str = 'circular',
                 p_agent_dropout: float = 0.,
                 requires_grad: bool = True,
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

        super().__init__()
        # Determine kernel sizes:
        # - First apply kernels preserving obs shape
        # - Lastly apply the kernel mapping convolved features to actions
        num_kernels = len(kernel_sizes)
        input_channels = [num_obs_channels] * num_kernels
        kernel_channels = [num_obs_channels] * (num_kernels-1) + [num_act_channels]

        kernels = [
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding='same',
                padding_mode=boundary,
                bias=False

            )
            for in_chans, kernel_size, out_chans
            in zip(input_channels, kernel_sizes, kernel_channels)
        ]
        # TODO: add fully connected (ie channel info exchange) before final output
        # NB: notice the last activation function
        # for normalizing the outputs into [-1, 1] range
        kernels.append(nn.Tanh())
        # Dropout for breaking synchrony between agent actions
        # TODO: need 1d dropout along channel dim
        # Channel-dropout is possible only with nn.Dropout1d and shape (W, H, C)
        # TODO: move dropout to Agent over 2d tensor (Batch, Agents, Channels)
        self.agent_dropout = nn.Dropout(p=p_agent_dropout)
        # Final model
        self.kernels = nn.Sequential(*kernels)
        self.requires_grad_(requires_grad)

    def init_weights(self):
        for kernel in self.kernels:
            if hasattr(kernel, 'weight'):
                xavier_uniform(kernel.weight)

    def forward(self, input: th.Tensor) -> th.Tensor:
        sense_transform = self.kernels(input)
        # NB: dropout can have no effect if it drops only dead cells
        output_shape = input.shape[2:]
        dropout_mask = self.agent_dropout(th.ones(output_shape))
        sense_transform *= dropout_mask
        return sense_transform


class NeuralAutomataAgent(Agent, nn.Module):
    """Agent with action mapping based on a neural model."""
    def __init__(self,
                 scale: float = 0.1,
                 deposit: float = 1.0,
                 with_agent_channel: bool = True,
                 initial_obs: Optional[ObsType] = None,
                 **model_kwargs,
                 ):
        super().__init__()
        self.obs_channels = list(DataChannels.medium) \
            if with_agent_channel else list(DataChannels.medium[1:])
        self.model = ConvolutionModel(num_obs_channels=len(self.obs_channels),
                                      num_act_channels=len(DataChannels.actions),
                                      **model_kwargs)
        # And these are coefficients for later scaling in forward() step
        self.action_coefs = np.array([scale, scale, deposit]).reshape((-1, 1))
        self._sense_output = initial_obs[1] \
            if initial_obs else np.ones((1, 1, 3))

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        idx = AgentIndexer(medium.shape[1:], agents)

        # Transform to Tensor, call internal tensor model, transform back to XArray
        sense_tensor = self.medium2tensor(self.obs_channels, medium)
        sense_tensor = self.model(sense_tensor)
        self._sense_output = self.tensor2medium(medium, sense_tensor)

        # Get per-agent actions from sensed environment transformed by reception model
        per_agent_output = idx.field_by_agents(self._sense_output, only_alive=False)
        # Rescale output actions
        per_agent_output = self._rescale(per_agent_output)
        # Build action XArray
        action = DataInitializer.init_action_for(agents, per_agent_output)
        return action

    def render(self) -> Sequence[np.ndarray]:
        data = self._sense_output.to_numpy()
        # Move channel dimension as the last dim
        rgb_channels = np.moveaxis(data, 0, -1)
        return [rgb_channels]

    def _rescale(self, per_agent_output: np.ndarray):
        # per_agent_output[:2, :] = (per_agent_output[:2, :] - .5) * 2.0
        per_agent_output *= self.action_coefs
        return per_agent_output

    @staticmethod
    def medium2tensor(channels, medium: MediumType) -> th.Tensor:
        # Select required channels
        sense_input: xa.DataArray = medium.sel(channel=channels)
        # Build Tensor with the shape in torch format:
        # (channels, width, height)
        sense_input_tensor = th.as_tensor(sense_input.values, dtype=th.float32)
            # .movedim([0, 1, 2], [2, 0, 1])
        # Workaround: reshape to 4D tensor for 'circular' padding
        # details: https://codesti.com/issue/pytorch/pytorch/95320
        shape4d = (1, *sense_input_tensor.shape)
        sense_input_tensor = th.reshape(sense_input_tensor, shape4d)
        return sense_input_tensor

    @staticmethod
    def tensor2medium(input_medium: MediumType, tensor: th.Tensor) -> MediumType:
        # Strip extra dims and reshape back to internal format:
        # (channels, width, height) -> (width, height, channels)
        tensor = tensor.squeeze() #.movedim([0, 1, 2], [1, 2, 0])
        # sense_transform = xa.DataArray(data=tensor, coords=input_medium.coords)
        sense_transform = xa.DataArray(data=tensor.detach().numpy(), coords=input_medium.coords)
        return sense_transform
