from typing import Tuple, Sequence

from core.agent.base import Agent
from core.base_types import ActType, ObsType
from core.data_init import DataInitializer
from core.utils import AgentIndexer


class EvoAgent(Agent):
    """Agent learning using Evolutionary Strategies"""

    def __init__(self,
                 kernel_sizes: Sequence[int] = (3,),
                 deposit: float = 0.):
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
        self._model = lambda x: x

    def forward(self, obs: ObsType) -> ActType:
        agents, medium = obs
        idx = AgentIndexer(medium.shape[1:], agents)

        sensed_channels = ('env_food', 'chem1')
        sense_input = medium.sel(channel=sensed_channels)
        sense_transform = self._model(sense_input)

        per_agent_output = idx.field_by_agents(sense_transform,
                                                  only_alive=False)
        action = DataInitializer.init_action_for(agents, per_agent_output)
        return action
