import logging
import sys
from pathlib import Path

from remote_handler.scripts.input_models import MaceFinetuningInput
from remote_handler.scripts.mace.mace_utils import (
    combine_databases,
    maybe_convert_to_jax,
    run_training_from_config,
    split_db,
)

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_train.json"
input_config = MaceFinetuningInput.from_file(INPUTFILE)
# Validate against calcpath so in-bundle paths (database_files, config_file,
# foundation_model) are checked for existence here, before training starts.
input_config.validate(calcpath)

# (paths in JSON are relative to calcpath, e.g. "../../db.extxyz")
database_path = combine_databases(input_config.database_files, calcpath)

config_path = calcpath / "mace_config.yaml"

split = split_db(
    database_path,
    calcpath / "db.extxyz",
    do_return_path=True,
    validation_fraction=input_config.validation_fraction,
    seed=input_config.seed,
)
train_path = split[0]
valid_path = split[1]

run_training_from_config(
    config_file=config_path,
    user_config_file=calcpath / input_config.config_file,
    model_name=input_config.model_name,
    seed=input_config.seed,
    train_file=train_path,
    valid_file=valid_path,
    train_runner=input_config.train_runner,
    storage_dir=calcpath,
    logger=logging.getLogger(),
    foundation_model=str((calcpath / input_config.foundation_model).resolve()),
)

# Optionally convert the trained torch model to a mace-jax bundle, in the same
# job/env (so torch-mace + mace-jax must both be installed there).
maybe_convert_to_jax(
    calcpath,
    input_config.model_name,
    input_config.convert_to_jax,
    input_config.jax_output_format,
    logger=logging.getLogger(),
)

# Success marker checked by ArrayScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("MACE finetuning finished")
