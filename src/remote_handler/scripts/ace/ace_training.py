import logging
import sys
from pathlib import Path

from remote_handler.scripts.ace.ace_utils import (
    build_ace_dataframe,
    run_ace_training,
)
from remote_handler.scripts.common.atoms_data import combine_databases
from remote_handler.scripts.input_models import AceTrainingInput

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_train.json"
input_config = AceTrainingInput.from_file(INPUTFILE)
# Validate against calcpath so in-bundle paths (database_files, config_file)
# are checked for existence here, before training starts.
input_config.validate(calcpath)

# (paths in JSON are relative to calcpath, e.g. "../../db.extxyz")
database_path = combine_databases(input_config.database_files, calcpath)

# pacemaker wants a pickled pandas dataframe, not extxyz directly.
dataframe_path = calcpath / "ace_train_data.pckl.gzip"
build_ace_dataframe(database_path, dataframe_path)

config_path = calcpath / "ace_config.yaml"
model_path = calcpath / "models" / f"{input_config.model_name}.yaml"

run_ace_training(
    config_file=config_path,
    user_config_file=calcpath / input_config.config_file,
    seed=input_config.seed,
    data_file=dataframe_path,
    storage_dir=calcpath,
    output_path=model_path,
    train_runner=input_config.train_runner,
    validation_fraction=input_config.validation_fraction,
    initial_potential=(
        calcpath / input_config.initial_potential
        if input_config.initial_potential
        else None
    ),
    logger=logging.getLogger(),
)

# Success marker checked by ArrayScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("ACE training finished")
