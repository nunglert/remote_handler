"""Example 3 -- Sampling configurations with torch-sim MD.

Configure a ``TorchSimSampler`` (parallel-tempering MD driven by a MACE model)
and run one sampling step to produce new candidate configurations.

Included verbatim by ``docs/source/examples.rst`` and signature-checked by
``test/test_examples.py``.
"""

from pathlib import Path

import remote_handler
from remote_handler.core import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows.samplers import TorchSimSampler

SCRIPTS = Path(remote_handler.__file__).parent / "scripts"

if __name__ == "__main__":
    config = ClusterConfig.from_name("datalab")
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    sampler = TorchSimSampler(
        remote_fs=remote_fs,
        cluster_config=config,
        sampling_script=str(SCRIPTS / "torch_sim" / "torch_sim_sampling.py"),
        founder_path="inputs/founder.extxyz",
        ensemble="nvt",
        temperatures=[300.0, 450.0, 600.0],
        target_temperature=300.0,
        model_paths=["mace_model.model"],
        n_steps_pt=200,
        device="cuda",
    )

    # Run MD from the current model and collect the new candidate configs.
    sampler.update(Path("md_run/0/sampler"), ["mace_model.model"])
    new_configs = sampler.get_sampled_configs()
