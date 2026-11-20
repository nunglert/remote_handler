"""Example 8 -- Structural analysis of a trajectory on a remote cluster.

Shard a set of trajectories, compute the radial distribution function and the
Steinhardt order parameters q4/q6 for every configuration as an array job on a
remote cluster, and collect the annotated configurations back -- one job, the
trajectory uploaded once, both analyses run back to back on each shard.

The remote conda env needs remote_handler with the ``steinhardt`` extra
(``jax``); RDF needs only ASE.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

import numpy as np
from ase.io import read

from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.processing import (
    RDFProcessor,
    RemoteTrajectoryProcessor,
    SteinhardtProcessor,
)

if __name__ == "__main__":
    config = ClusterConfig.from_name("lugus")
    remote_fs = RemoteFileSystem(remote_alias="lugus")

    handler = RemoteTrajectoryProcessor(
        traj_files=["md_300K.extxyz", "md_600K.extxyz"],
        cluster_config=config,
        remote_fs=remote_fs,
        n_processors=1,
        shard_size=200,  # configurations per array task
        savepath="/scratch/me/analysis",
        num_parallel=10,  # array tasks running at once
        # Steps run in this order on every shard.
        processors=[
            RDFProcessor(cutoff=6.0, nbins=300),
            SteinhardtProcessor(r_cut=3.0, l_values=(4, 6)),
        ],
        tag="ana",
    )

    # Stage the shards, submit the array, wait, then collect and write the
    # results next to the inputs as ana_md_300K.extxyz / ana_md_600K.extxyz.
    handler.run()

    # Per-frame results are in atoms.info, per-atom ones in atoms.arrays.
    for at in read("ana_md_300K.extxyz", index=":3"):
        r = np.asarray(at.info["r_rdf"])
        g = np.asarray(at.info["rdf_total"])
        print(
            f"g(r) peak at {r[g.argmax()]:.2f} A,",
            f"<q6> = {at.info['q6_mean']:.3f},",
            f"per-atom q6: {at.arrays['q6'].shape}",
        )
