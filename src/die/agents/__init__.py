from die.agents.base import Agent
from die.agents.gradient import GradientAgent, PhysarumAgent
from die.agents.neural import ConvolutionModel, NeuralAutomataAgent, TorchAgent
from die.agents.static import BrownianAgent, ConstAgent

__all__ = [
    "Agent",
    "BrownianAgent",
    "ConstAgent",
    "ConvolutionModel",
    "GradientAgent",
    "NeuralAutomataAgent",
    "PhysarumAgent",
    "TorchAgent",
]
