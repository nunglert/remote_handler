"""
Active Learning Workflow Components.

This module provides workflow components for active learning pipelines:

- Evaluators: Ground truth evaluation (e.g., DFT with VASP)
- Samplers: Configuration sampling (e.g., LAMMPS MD, torch_sim)
- Trainers: Model training (e.g., MACE models)

Required Dependencies:

- ase: Atomic simulation environment (atoms handling)
- numpy: Numerical operations

Each component can run in stub mode for testing without actual execution.
"""

from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.evaluators import (
    AseCalculatorEvaluator,
    EMTEvaluator,
    ORCAEvaluator,
    VASPEvaluator,
)
from remote_handler.workflows.samplers import LammpsSampler, TorchSimSampler
from remote_handler.workflows.staged_active_learning import (
    InputSource,
    SelectorActor,
    Stage,
    StagedActiveLearningHandler,
)
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
    NeuralilILTrainer,
)

__all__ = [
    "AseCalculatorEvaluator",
    "VASPEvaluator",
    "ORCAEvaluator",
    "EMTEvaluator",
    "LammpsSampler",
    "TorchSimSampler",
    "MaceFinetuningTrainer",
    "MaceModelTrainer",
    "NeuralilILTrainer",
    "StagedActiveLearningHandler",
    "Stage",
    "InputSource",
    "Selector",
    "SelectorActor",
]
