
from Env import *
from typing import *

import numpy as np
import torch as th
import torch.nn as nn

from core.data_utilities import index_select, np_mask_duplicates


class AgentNet(nn.Module):

    def __init__(self, obs_shape: Tuple[int, int, int],
                 kernel_dim: int, num_kernels: int = 2, boundary: str = 'circular',
                 p_agent_dropout: float = 0.5
                 ):
        super().__init__()
        self.obsspec = ObsSpec.from_obs(obs_shape)
        self.boundary = boundary
        assert boundary in {'circular', 'zero'}
        assert num_kernels > 0

        kernel_sz = (kernel_dim,) * self.obsspec.ncoords
        # first apply kernels preserving obs shape
        # lastly apply the kernel mapping convolved features to actions
        out_channels_ls = [self.obsspec.nchans] * (num_kernels-1) + [Actions.dim]
        # TODO: do I do some sample eg from Normal distrib? or deterministic policy?
        self.kernels = nn.Sequential(*[
            nn.Conv2d(
                in_channels=self.obsspec.nchans,
                out_channels=out_chans,
                kernel_size=kernel_sz,
                padding='same',
                padding_mode=boundary,
                bias=True
            )
            for out_chans in out_channels_ls
        ],  nn.Sigmoid())
        # NB: notice the last activation function for normalizing the outputs into [0,1] range
        # And these are coefficients for later scaling in forward() step
        self.actions_coefs = np.array(Actions.channel_ranges()[1])
        # Dropout for breaking synchrony between agent actions
        self.agent_dropout = nn.Dropout(p=p_agent_dropout)

    def __init_weights(self):
        # TODO impl
        pass

    def forward(self, obs: th.Tensor, agents: AgentArrayWrapper) -> Actions:
        # NB: agents is a variable-length array
        assert obs.shape == self.obsspec.shape # TODO: here in principle can have some batch?
        assert obs.dtype == th.float32 # TODO: here possibly can somewhere setup up global dtype as param

        # The problem is not to get sense offset indices -- that simple mapping agents.to_indices(with_offset)
        #   and here I get unique mapping AgentIdx -> (iposX,iposY).
        #   but I need inverse mapping (iposX,iposY) -> AgentIdx, which is non-unique.
        #   But! numpy/torch allow non-unique indexing. So that's ok.
        # TODO: I just need ensuring  that no operations fail or complicate shit
        #  because of several agents in the same cell.

        # Concerns:
        # - Agents in the same cell result in the same action!
        # Need some assumption or restriction to forbid agents to "merge" in this way. Antialiasing of agents:
        # - Only one ell among aliased is moved
        # - Updates or the policy is inherently stochastic AND stochastic sampling is applied AFTER inverse mapping.

        agent_positions = agents.to_indices(self.obsspec.grid, with_offset=True)
        # Zero-out cells trying to get to the same place
        # In fact, because of stochastic dropout, antialiasing will be automatic, all ok!
        # TODO: maybe remove, because duplicates are masked out in env.agents.move
        antialias_mask = np_mask_duplicates(agent_positions, axis=0)

        # Main logic

        all_agents_effect_on_grid = self.kernels(obs)
        all_agents_effect = index_select(all_agents_effect_on_grid, th.as_tensor(agent_positions))

        # Mask effects from dropped-out and non-alive agents
        active_agents_mask = th.bitwise_not(agents.mask_dead() | antialias_mask)
        # NB: dropout can have no effect if it drops only dead cells
        active_agents_mask = self.agent_dropout(active_agents_mask)

        assert all_agents_effect.shape == (len(agents), self.actions.dim)

        return Actions(self._renormalize_output(all_agents_effect), active_agents_mask)

    def _renormalize_output(self, agents_effect: th.Tensor) -> th.Tensor:
        return agents_effect * self.actions_coefs
