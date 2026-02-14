"""CMA-ES evolution sampling script."""

import pathlib
import sys
from pathlib import Path

from clinamen2.utils.file_handling import CMAFileHandler

from remote_handler.scripts.cmaes_utils import run_evolution
from remote_handler.scripts.input_models import CmaesSamplingInput

# Load and validate input configuration
calcpath = Path(sys.argv[1]).parent
INPUTFILE = Path(sys.argv[1])
input_config = CmaesSamplingInput.from_file(INPUTFILE)
input_config.validate()

if input_config.continue_evolution:
    try:
        handler = CMAFileHandler(
            target_dir=pathlib.Path.cwd() / input_config.label,
            label=input_config.label,
        )
        (
            loaded_parameters,
            _,
            _,
            information,
        ) = handler.load_evolution()
    except Exception:
        print(
            f"No data found for label {input_config.label} in "
            f"{pathlib.Path.cwd() / input_config.label}."
        )
        raise
    generation = input_config.generation_checkpoint
    if generation is None:
        try:
            generation = information["last_gen"]
        except KeyError:
            print(
                "No field 'last_gen' in evolution data. "
                "Please provide 'generation_checkpoint' in input.json."
            )
            raise
    try:
        loaded_generation = handler.load_generation(generation=generation)
    except Exception:
        print(
            f"No data found for generation {generation} in "
            f"{pathlib.Path.cwd() / input_config.label}."
        )
        raise
    print(
        f"Continue evolution '{input_config.label}' from generation {generation}."
    )
else:
    loaded_parameters = None
    loaded_generation = None

print("\n----------------------------")
print(
    f"Evolution {input_config.label} for {input_config.generations} generations."
)
evolution_args = {
    "founder": input_config.founder,
    "models": input_config.models,
    "run_label": input_config.label,
    "generations": input_config.generations,
    "run_seed": input_config.random_seed,
    "initial_step_size": input_config.step_size,
    "pop_size": input_config.pop_size,
    "save_nth": input_config.save_nth,
    "batch": input_config.batch,
    "loss_mode": input_config.loss_mode,
    "loss_threshold": input_config.loss_threshold,
    "scaled_center": input_config.scaled_center,
    "radius": input_config.radius,
    "c_r": input_config.c_r,
    "continue_evolution": input_config.continue_evolution,
    "continue_with_parameters": loaded_parameters,
    "continue_from_checkpoint": loaded_generation,
    "repository_path": input_config.repository_path,
}
last_gen, _, _, _ = run_evolution(**evolution_args)
