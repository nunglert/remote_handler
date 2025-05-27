import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from re import T
from typing import Any, Optional, Union

from remote_handler.atoms_handler import RemoteAtomsHandler
from remote_handler.core import cluster_handler, file_handler, tasks

logger = logging.getLogger(__name__)


class VASPHandler(RemoteAtomsHandler):
    def __init__(
        self,
        traj_files: str,
        remote_fs: file_handler.FileSystemHandlerBase,
        savepath: str,
        num_parallel: int,
        loop_time: float = 30,
        jobname: str = "opt",
        every: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        local_savepath: Optional[os.PathLike] = None,
        logfile: str = "dft.log",
        collect_only: bool = False,
        tag: str = "dft",
        python_script_path: str = "/home/nico.unglert/code/remote_handler/src/remote_handler/scripts/dft_singlepoint.py",
        stub: bool = False,
        launcher: str = "mpiexec -np",
        slurm_header_add: str = "#SBATCH --time=0-02:00:00\nmodule purge\nmodule load vasp/6.4.2_intel\nsource /opt/intel/oneapi/setvars.sh --force\nexport VASP_PP_PATH=/home/nunglert/cellar/vasp/paw",
        vasp_binary: str = "vasp_std",
        n_processors: int = 8,
        incar_kwargs: dict = {
            # "istart": 0,
            # "icharg": 2,
            "xc": "R2SCAN",
            "encut": 300,
            "prec": "Normal",
            "ismear": -1,
            "sigma": 0.025,
            "ibrion": -1,
            "ediff": 1e-5,
            # "isym": 0,
            "symprec": 1e-7,
            "lwave": False,
            "lcharg": False,
            "lreal": False,
            # "kspacing": 2.0
            "kspacing": 0.3,
        },
    ):
        #         slurm_header_add = """#SBATCH --time=0-02:00:00
        # module load vasp/6.4.2_intel
        # source /opt/intel/oneapi/setvars.sh --force
        # export VASP_PP_PATH=/home/nunglert/cellar/vasp/paw
        # """
        super().__init__(
            traj_files,
            remote_fs,
            1,  # shard_size
            savepath,
            num_parallel,
            loop_time,
            jobname,
            every,
            start,
            end,
            local_savepath,
            logfile,
            collect_only,
            tag,
            n_processors,
            slurm_header_add=slurm_header_add,
        )

        self.logger = tasks.SimpleLogger(logfile)
        self.json_args = {
            "STUB": stub,
            "INCAR_kwargs": incar_kwargs,
            "launch_command": f"{launcher} {n_processors} {vasp_binary}",
        }

        # Since we did not initialize the python_script_file, shared_ressources
        # and individual_ressources in parent class, we need to do it here!
        self.python_script_file = python_script_path

        self.shared_ressources = []

        if not collect_only:
            self.prepare_individual_data()

    def prepare_individual_data(self):
        config_files = []
        atoms_batches = []
        for calc_idx in range(self.num_array_tasks):
            json_config = deepcopy(self.json_args)

            # TODO: this is dirty, properly solve this
            json_config["DATABASE"] = f"dft.extxyz"

            config_files.append(
                tasks.DictJSON(json_config, "input_dft.json", self.remote_fs)
            )
            atoms_batches.append(
                tasks.AtomsBatch(
                    self.shards[calc_idx],
                    json_config["DATABASE"],
                    self.remote_fs,
                )
            )

        del self.shards

        self.individual_ressources = [config_files, atoms_batches]

    def finish_check(self, required_fraction: float = 0.8):
        assert self.all_calcpaths, "There are no calcpaths to check!"

        num_converged = 0
        for calcpath in self.all_calcpaths:
            if self.remote_fs.exists(Path(calcpath) / "is_finished.info"):
                num_converged += 1

        finished_fraction = num_converged / len(self.all_calcpaths)
        if finished_fraction < 0.5:
            logger.warning("More than 50% of the submitted runs failed..")
        return finished_fraction > required_fraction

    def run(self):
        self.allocate_data()
        self.submit()
        # self.collect_and_harvest()
        self.collect_results()
