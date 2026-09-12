"""Example 4 -- Ground-truth labelling with ORCA.

Run ORCA single-point calculations on a set of structures as a SLURM array
job on a remote cluster, then collect the labelled configurations -- the
same actor shape as Example 1 (VASP), swapping the DFT code.

remote_handler does not ship ORCA itself -- it must already be installed on
the target cluster, and ``orca_binary``/the cluster's ``environment["orca"]``
entry must point at it.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.evaluators import ORCAEvaluator

if __name__ == "__main__":
    config = ClusterConfig.from_name("dirona")
    remote_fs = RemoteFileSystem(remote_alias="dirona")

    evaluator = ORCAEvaluator(
        cluster_config=config,
        remote_fs=remote_fs,
        num_parallel=10,
        orcasimpleinput="PBE def2-SVP",
        n_processors=8,
    )

    # Label the sampler's candidate configs; calcdir is this iteration's
    # work dir.
    evaluator.update(Path("al_run/0/dft"), Path("sampled_configs.extxyz"))
    labelled = evaluator.get_evaluated_configs()
