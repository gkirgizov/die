"""Evolutionary training of neural agents.

The concrete searcher backend (``die.train.evotorch_backend``) requires
the ``train`` extra; everything else here depends only on core deps.
"""

from die.train.loggers import HistoryLogger, MlflowTrainLogger, StdoutTrainLogger, TrainLogger
from die.train.searcher import GenStats, SearcherBackend
from die.train.trainer import EvolutionTrainer, TrainConfig

__all__ = [
    "EvolutionTrainer",
    "GenStats",
    "HistoryLogger",
    "MlflowTrainLogger",
    "SearcherBackend",
    "StdoutTrainLogger",
    "TrainConfig",
    "TrainLogger",
]
