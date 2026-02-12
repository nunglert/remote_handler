"""
Active Learning Workflow Components.

This module provides workflow components for active learning pipelines:
- Evaluators: Ground truth evaluation (e.g., DFT with VASP)
- Samplers: Configuration sampling (e.g., LAMMPS MD, Nested Sampling)
- Trainers: Model training (e.g., MACE models)

Required Dependencies:
- ase: Atomic simulation environment (atoms handling)
- numpy: Numerical operations

Optional Dependencies:
- jaxnest: Nested sampling (for NSTrajectorySampler, MACETrainer)
  Only required when not using stub mode.

Each component can run in stub mode for testing without actual execution.
"""

from remote_handler.workflows.active_learning.evaluators import VASPEvaluator
from remote_handler.workflows.active_learning.samplers import (
    LammpsSampler,
    NSTrajectorySampler,
)
from remote_handler.workflows.active_learning.trainers import (
    MaceModelTrainer,
    MACETrainer,
    NeuralilILTrainer,
)

__all__ = [
    "VASPEvaluator",
    "LammpsSampler",
    "NSTrajectorySampler",
    "MaceModelTrainer",
    "MACETrainer",
    "NeuralilILTrainer",
]
