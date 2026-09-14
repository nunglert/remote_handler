"""Example 1 -- DFT single-point labelling with VASP.

Run VASP single-point calculations on a set of structures as a SLURM array job
on a remote cluster, then collect the labelled configurations.

remote_handler does not ship VASP itself (a licensed, external code) -- it
must already be installed on the target cluster, and ``vasp_binary``/the
cluster's ``environment["vasp"]`` entry must point at it.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.dft_runner import VASPHandler

if __name__ == "__main__":
    config = ClusterConfig.from_name("dirona")
    remote_fs = RemoteFileSystem(remote_alias="dirona")

    handler = VASPHandler(
        traj_files=["selected_structures.extxyz"],
        cluster_config=config,
        remote_fs=remote_fs,
        savepath="/scratch/me/vasp",
        num_parallel=10,
        n_processors=16,
        incar_kwargs={
            "xc": "PBE",
            "encut": 400,
            "ediff": 1e-6,
            "ismear": 0,
            "sigma": 0.05,
            "kpts": [3, 3, 1],
        },
    )

    # Stage inputs, submit the array, wait, and collect the labelled configs.
    handler.run()
