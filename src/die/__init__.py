"""Distributed Intelligence Environment (DIE).

An Artificial Life project: learning cellular automata agents
under environmental pressures for foraging, feeding and not-dying.
"""

__version__ = "0.1.0"

from die.agents.base import Agent
from die.agents.gradient import GradientAgent, PhysarumAgent
from die.agents.neural import ConvolutionModel, NeuralAutomataAgent, TorchAgent
from die.agents.static import BrownianAgent, ConstAgent
from die.env import Dynamics, Env
from die.registry import AgentId, DynamicsId, load_agent, make_agent, make_dynamics, make_env
from die.sim import MetricsObserver, RunResult, SimObserver, Simulation, StepEvent, TqdmObserver
from die.types import ActType, AgtType, DataChannels, MediumType, ObsType

__all__ = [
    "ActType",
    "Agent",
    "AgentId",
    "AgtType",
    "BrownianAgent",
    "ConstAgent",
    "ConvolutionModel",
    "DataChannels",
    "Dynamics",
    "DynamicsId",
    "Env",
    "GradientAgent",
    "MediumType",
    "MetricsObserver",
    "NeuralAutomataAgent",
    "ObsType",
    "PhysarumAgent",
    "RunResult",
    "SimObserver",
    "Simulation",
    "StepEvent",
    "TorchAgent",
    "TqdmObserver",
    "load_agent",
    "make_agent",
    "make_dynamics",
    "make_env",
]
