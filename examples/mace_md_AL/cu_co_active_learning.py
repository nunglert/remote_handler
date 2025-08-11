import logging
import os
from pathlib import Path
from typing import Optional

import ase.io
from ase import Atoms

from remote_handler.core.active_learning_base import (
    GroundTruthEvaluator,
    ModelTrainer,
    Sampler,
)
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import (
    ArrayScriptTask,
    DictJSON,
    SingleScriptTask,
)
from remote_handler.core.utils import grep
from remote_handler.dft_runner import VASPHandler

logger = logging.getLogger(__name__)


class VASPEvaluator(GroundTruthEvaluator):
    def __init__(
        self,
        singlepoint_script: os.PathLike,
        remote_fs: FileSystemHandlerBase,
        num_parallel: int,
        remote_savepath: Optional[os.PathLike] = None,
        launcher: str = "mpirun -n",
        incar_kwargs: dict = {
            "xc": "PBE",
            "encut": 300,
            "prec": "Normal",
            "ismear": -1,
            "sigma": 0.025,
            "ibrion": -1,
            "ediff": 1e-5,
            "symprec": 1e-5,
            "lwave": False,
            "lcharg": False,
            "lreal": False,
            "kspacing": 2.0,
        },
        stub: bool = False,  # just for testing
        loop_time: float = 30,
        required_fraction: float = 0.5,
        n_processors: int = 2,
    ):
        super().__init__()
        self.singlepoint_script = singlepoint_script
        self.incar_kwargs = incar_kwargs
        self.remote_fs = remote_fs
        self.num_parallel = num_parallel
        self.stub = stub
        self.launcher = launcher
        self.loop_time = loop_time
        self.required_fraction = required_fraction
        self.remote_savepath = Path(remote_savepath)
        self.n_processors = n_processors

    def update(
        self, calcdir: os.PathLike, collected_file: os.PathLike, **kwargs
    ) -> None:
        self.dft_dir = Path(calcdir) / "dft"
        self.results_dir = self.dft_dir / "results"
        os.makedirs(self.dft_dir, exist_ok=True)
        it = str(calcdir).split("/")[-1]

        self.vasp_handler = VASPHandler(
            loop_time=self.loop_time,
            jobname="dft",
            every=1,
            start=None,
            end=None,
            logfile="dft.log",
            tag="dft",
            stub=self.stub,
            collect_only=False,
            remote_fs=self.remote_fs,
            savepath=self.remote_fs.slurm_config["savepath"],
            local_savepath=self.results_dir,
            traj_files=[collected_file],
            python_script_path=self.singlepoint_script,
            num_parallel=self.num_parallel,
            n_processors=self.n_processors,
            incar_kwargs=self.incar_kwargs,
            launcher=self.launcher,
        )

    def get_evaluated_configs(self) -> list[Atoms]:
        self.vasp_handler.run()

        if not self.vasp_handler.finish_check(self.required_fraction):
            raise Exception(
                f"Less than {self.required_fraction * 100}% ran successfully. "
                "Something probably went wrong. Otherwise, lower the "
                "required_fraction parameter."
            )

        all_ats = []
        for calcpath in grep(self.results_dir / "calcdirs", ""):
            try:
                at = ase.io.read(Path(calcpath) / "vasprun.xml")
                all_ats.append(at)
            except:
                logger.info(f"VASP run failed for {calcpath=}")

        return all_ats


class MaceModelTrainer(ModelTrainer):
    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        training_script: os.PathLike,
        template_path: os.PathLike,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 5,
    ):
        super().__init__()
        self.training_script = training_script
        self.remote_fs = remote_fs
        self.template_path = template_path
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time

    def update(
        self, calcdir: os.PathLike, db_files: list[os.PathLike], **kwargs
    ) -> None:
        self.train_dir = Path(calcdir) / "train"
        self.results_dir = self.train_dir / "results"
        os.makedirs(self.train_dir, exist_ok=True)

        script_input_shared = {
            "database_files": [Path(p).name for p in db_files],
            "template_file": Path(self.template_path).name,
            "model_name": "Test_Model",
            "train_runner": "/home/e12218283/miniconda3/envs/bsc_base/lib/python3.10/site-packages/mace/cli/run_train.py",
        }

        individual_resources = [
            [
                DictJSON(
                    content=script_input_shared | {"seed": i + 10},
                    filename="input_train.json",
                    remote_fs=self.remote_fs,
                )
                for i in range(5)
            ]
        ]

        self.training_task = ArrayScriptTask(
            local_savepath=self.results_dir,
            remote_fs=self.remote_fs,
            python_script_file=self.training_script,
            individual_ressources=individual_resources,
            shared_ressources=[self.template_path] + db_files,
            logfile="train.log",
            remote_savepath=self.remote_savepath,
            collect_only=False,
            num_parallel=5,
            loop_time=self.loop_time,
            jobname="train",
            n_processors=1,
            jobfile_name="run_train.sh",
        )

        self.training_task.allocate_data()

    def get_trained_model(self) -> os.PathLike:
        self.training_task.submit()
        self.training_task.collect_results()

        """models_dir = self.results_dir / 'models'
        os.makedirs(models_dir, exist_ok=True)
        for p in (results_dir / 'calcdirs').iterdir():
            if p.is_dir():
                self.local_fs.put(p / 'models/Test_Model_stagetwo.model', models_dir)"""

        return Path(self.training_task.local_savepath) / "train/results"
