import logging
import sys
from pathlib import Path

import ase.io

from remote_handler.scripts.common.atoms_data import combine_databases
from remote_handler.scripts.grace.grace_utils import (
    finalize_grace_artifact,
    run_grace_training,
    write_ase_standard_labels,
)
from remote_handler.scripts.input_models import GraceTrainingInput

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_train.json"
input_config = GraceTrainingInput.from_file(INPUTFILE)
# Validate against calcpath so in-bundle paths (database_files, config_file)
# are checked for existence here, before training starts.
input_config.validate(calcpath)

# (paths in JSON are relative to calcpath, e.g. "../../db.extxyz")
database_path = combine_databases(input_config.database_files, calcpath)

# gracemaker reads plain ASE energy/forces (via a real calculator), not
# remote_handler's REF_energy/REF_forces convention -- relabel first.
data_path = calcpath / "grace_train_data.extxyz"
write_ase_standard_labels(data_path, ase.io.read(database_path, index=":"))

config_path = calcpath / "grace_config.yaml"

run_grace_training(
    config_file=config_path,
    user_config_file=calcpath / input_config.config_file,
    seed=input_config.seed,
    data_file=data_path,
    storage_dir=calcpath,
    train_runner=input_config.train_runner,
    validation_fraction=input_config.validation_fraction,
    logger=logging.getLogger(),
)

finalize_grace_artifact(calcpath, input_config.seed, input_config.model_name)

# Success marker checked by ArrayScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("GRACE training finished")
