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

- jaxnest: Nested sampling (for NSTrajectorySampler), only required when not
  using stub mode.

Each component can run in stub mode for testing without actual execution.
"""

from remote_handler.workflows.evaluators import VASPEvaluator
from remote_handler.workflows.samplers import (
    LammpsSampler,
    NSTrajectorySampler,
    TorchSimSampler,
)
from remote_handler.workflows.trainers import (
    MaceFinetuningTrainer,
    MaceModelTrainer,
    NeuralilILTrainer,
)

__all__ = [
    "VASPEvaluator",
    "LammpsSampler",
    "NSTrajectorySampler",
    "TorchSimSampler",
    "MaceFinetuningTrainer",
    "MaceModelTrainer",
    "NeuralilILTrainer",
]
