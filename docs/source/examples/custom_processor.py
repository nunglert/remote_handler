"""A custom processor: per-atom coordination numbers.

Shows the full contract for a processor defined outside remote_handler, all in
one module: the arg schema, the cluster-side ``run``, the host-side
``Processor`` spec, and the ``register`` call. In a real project this module
lives in your own package (e.g. ``my_project/processors/coordination.py``),
which must also be installed in the cluster's conda env.

Included verbatim by ``docs/source/processing.rst`` and run through the real
shard driver by ``test/test_processing_docs.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase.neighborlist import neighbor_list

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing import Processor, register
from remote_handler.scripts.input_models import ScriptInput


# 1. Arg schema: one source of truth for the step's arguments. The host fills
#    it in build_args; the driver validates against it before any compute.
@dataclass
class CoordinationArgs(ScriptInput):
    cutoff: float

    def validate(self, root: Optional[Path] = None) -> None:
        if not isinstance(self.cutoff, (int, float)) or self.cutoff <= 0:
            raise ScriptInputError("coordination: 'cutoff' must be > 0")


# 2. Cluster-side compute: takes and returns the shard's list of Atoms.
#    Annotate in place: per-frame values in atoms.info, per-atom values in
#    atoms.arrays. Keep heavy imports inside run so the host stays light.
def run(atoms_list, args, calcpath=None):
    for at in atoms_list:
        centers = neighbor_list("i", at, args["cutoff"])
        cn = np.bincount(centers, minlength=len(at))
        at.arrays["coordination"] = cn
        at.info["coordination_mean"] = float(cn.mean())
    return atoms_list


# 3. Host-side spec. `name` must match the register() call below. Because the
#    class lives in the same module as that call, registration_module defaults
#    to this module, and the driver imports it on the cluster automatically.
class CoordinationProcessor(Processor):
    name = "coordination"

    def __init__(self, cutoff: float = 3.0):
        self.cutoff = cutoff

    def build_args(self, savepath: str) -> dict:
        return CoordinationArgs(cutoff=self.cutoff).to_dict()


# 4. Registration runs at import time (on the cluster, when the driver imports
#    this module). It must be idempotent: re-importing just re-registers.
register("coordination", run, schema=CoordinationArgs)
