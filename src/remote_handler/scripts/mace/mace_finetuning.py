import logging
import os
import sys
from pathlib import Path

from remote_handler.scripts.input_models import MaceFinetuningInput
from remote_handler.scripts.mace.mace_utils import (
    combine_databases,
    convert_torch_model_to_jax,
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

# Optionally convert the trained torch model to a mace-jax .pkl bundle, in the
# same job/env (so torch-mace + mace-jax must both be installed there).
if input_config.convert_to_jax:
    torch_model_path = calcpath / "models" / f"{input_config.model_name}.model"
    assert os.path.exists(
        torch_model_path
    ), f"Trained model not found for conversion: {torch_model_path}"
    jax_model_path = convert_torch_model_to_jax(
        torch_model_path,
        output_path=calcpath / "models" / f"{input_config.model_name}-jax.pkl",
    )
    logging.getLogger().info(
        f"Converted torch model to mace-jax bundle: {jax_model_path}"
    )
