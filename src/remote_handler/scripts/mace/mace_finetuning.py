import logging
import os
import sys
from pathlib import Path

from remote_handler.scripts.input_models import MaceFinetuningInput
from remote_handler.scripts.mace.mace_utils import (
    combine_databases,
    run_training_from_config,
    split_db,
)

# Load and validate input configuration
calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_train.json"
input_config = MaceFinetuningInput.from_file(INPUTFILE)
input_config.validate()

for f in input_config.database_files:
    assert os.path.exists(calcpath / f), f"missing database file {f}"
assert os.path.exists(
    calcpath / input_config.config_file
), "Missing config_file"
assert os.path.exists(
    calcpath / input_config.foundation_model
), "Missing foundation_model"

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
