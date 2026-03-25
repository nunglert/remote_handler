"""Evolution example: Si bulk with NNFF"""

import copy
import json
import pathlib
import sys
from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import torch
from ase.io import write
from clinamen2.cmaes.cmaes_criteria import (
    EqualFunValuesCriterion,
    TolXUpCriterion,
)
from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_resample_and_evaluate,
    create_sample_and_sequential_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import (
    CriteriaOr,
    StaleLossCriterion,
    StaleStepCriterion,
)
from clinamen2.utils.file_handling import CMAFileHandler, JSONEncoder
from clinamen2.utils.script_functions import cma_parser, cma_setup
from clinamen2.utils.structure_setup import FilterEvalWorkflow
from clinamen2.utils.structure_setup import (
    bias_covariance_matrix_r as bias_covariance_matrix,
)
from clinamen2.utils.structure_setup import (
    create_filter_eval_workflow,
    prepare_dof_and_pipeline,
)
from mace.calculators import MACECalculator
from tqdm import tqdm

ADVERSARIAL = "adversarial"
ENERGY_ONLY = "energy-only"
WITH_UNCERTAINTIES = "with-uncertainties"
LOSS_THRESHOLD = "loss-threshold"

FULL_DOF = "full-dof"
GLIDE_DOF = "glide-dof"
MIRROR_DOF = "mirror-dof"

RANDOM_FOUNDER = "random"

SORTED_ELEMENTS = sorted(["O", "Sr", "Ti"])
STANDARD_MODE = "standard"
N_ENSEMBLE = 5
DEFAULT_FOUNDER_PATH = pathlib.Path.cwd().parents[0] / "data" / "founders"
DEFAULT_FOUNDER = DEFAULT_FOUNDER_PATH / "POSCAR_sto110_2x2"
DEFAULT_MODEL = (
    pathlib.Path.cwd().parents[0]
    / "mace/mace-4x1m_2x2_2x3ab/checkpoints/MACE-4x1m_2x2_2x3ab_run-123_swa.model"
)


def create_lossthreshold_filter(
    loss_threshold: float = None,
    loss_threshold_exception: Exception = BaseException,
) -> Callable:
    """Create function to filter configurations.

    The filter is applied with regards to an absolute loss threshold.
    This is a lower bound that may not be exceeded.
    "Poor man's min distance exception."

    Args:
        - loss_threshold: Absolute value of lower loss bound.
            Default is None, filters will be identity functions.
    """

    if loss_threshold is None:
        return lambda *x: x, lambda *x: x

    else:

        def batch_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ) -> Tuple[float, list]:
            for a in additional:
                check_val = a["loss"]
                if check_val < loss_threshold:
                    a["exception"] = loss_threshold_exception(
                        f"Loss {check_val} "
                        f"below threshold of {loss_threshold}."
                    )

            return loss, additional, inputs

        def single_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ):
            if loss < loss_threshold:
                additional["exception"] = loss_threshold_exception(
                    f"Loss {loss} " f"below threshold of {loss_threshold}."
                )

            return loss, additional, inputs

        return single_lossthreshold_filter, batch_lossthreshold_filter


class LossThresholdException(Exception):
    """Exception to be raised for loss below threshold.

    This exception is raised instead of checking
    min distances.
    """

    pass


# tuple input parser function
# no error handling
def parse_tuple(val_string: str):
    nruter = []
    vals = val_string.split(",")
    for val in vals:
        nruter.append(float(val))

    return np.asarray(nruter)


class InputPipeline(NamedTuple):
    """Combined pipelines for input transformation.

    Args:
        placeholder_pipeline: Function pipeline performing placeholder specific data
            transformation.
        clinamen_pipeline: Function pipeline performing Clinamen2 specific
            data transformation.
    """

    placeholder_pipeline: Callable = None  # nothing
    clinamen_pipeline: Callable = None  # dof -> structure


def create_input_pipeline(pipeline: InputPipeline) -> Callable:
    """Create combined pipeline.

    Args:
        pipeline: Function pipeline for input transformation.
    """

    def input_pipeline(
        dof: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Apply input pipeline."""
        transformed = copy.deepcopy(dof)
        transformed = transformed.reshape((-1,))

        for fun in pipeline[::-1]:
            transformed = fun(transformed) if fun is not None else transformed
        transformed.set_pbc([1, 1, 0])

        return transformed

    return input_pipeline


def create_batch_input_pipeline(pipeline: InputPipeline) -> Callable:
    """TODO: Refactor for this application.

    Args:
        pipeline: Function pipeline for batched input transformation.
    """

    def input_pipeline(
        dof: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Apply input pipeline."""
        input_values = copy.deepcopy(dof)
        transformed = []

        for t in input_values:
            for fun in pipeline[::-1]:
                t = fun(t) if fun is not None else t
                t.set_pbc([1, 1, 0])
            transformed.append(t)

        return transformed

    return input_pipeline


def create_mace_energy_single_and_batch(models):
    """Closures utilizing calculator for committee."""

    calcommittee = MACECalculator(
        model_paths=[model for model in models],
        device="cuda",
        default_dtype="float32",
    )

    if len(models) > 1:

        def mace_energies(atoms):
            """Return mean energy."""
            btoms = atoms.copy()
            btoms.calc = None
            btoms.calc = calcommittee
            btoms.get_potential_energies()
            energy = np.mean(btoms.calc.results["energies"])

            return energy

        mace_energy_single = mace_energies
    else:

        def mace_energy(atoms):
            """Return mean energy."""
            btoms = atoms.copy()
            btoms.calc = None
            btoms.calc = calcommittee
            energy = btoms.get_potential_energy()

            return energy

        mace_energy_single = mace_energy

    def mace_energy_batch(atoms_list):
        energies = []
        additional = []
        for atoms in atoms_list:
            # energy, _ = mace_energy_single(atoms)
            # # TODO: NU: why two quantities here?
            energy = mace_energy_single(atoms)
            energies.append(energy)
            additional.append({"loss": energy})  # may not be empty
        return energies, additional

    return mace_energy_single, mace_energy_batch


def run_evolution(
    founder: str,
    models: List[str],
    run_label: str = "",
    run_seed: int = 0,
    generations: int = 5,
    initial_step_size: float = 0.12,
    pop_size: int = None,
    batch: bool = False,
    save_nth: int = 1,
    loss_mode: str = "energy-only",
    loss_threshold: float = None,
    scaled_center: Tuple[float, float, float] = None,
    radius: float = None,
    c_r: float = None,
    continue_evolution: bool = False,
    continue_with_parameters: AlgorithmParameters = None,
    continue_from_checkpoint: Tuple[
        AlgorithmState, npt.ArrayLike, npt.ArrayLike, dict
    ] = None,
    repository_path: str = None,
) -> int:
    """Perform evolution with given parameters.

    Args:
        - founder_file: 'Random' or full path to POSCAR of founder structure.
        - models: Committee members.
        - run_label: Name of the evolution run. Also used to name the
            subfolder containing the results.
        - run_seed: Random number seed for initialization. Default is 0.
        - generations: Maximum number of generations to run. Default is 5.
        - initial_step_size: CMA-ES parameter step size. Default is 0.12.
        - batch: Boolean flag to switch between vmapped evaluation (True)
            and sequential evaluation (False).
        - save_nth: Every n-th generation will be saved as a json file. Default
            is 1.
        - loss_mode: Which loss function to use.
        - dof_mode: Full vs mirror vs glide.
        - scaled_center: Point to center the sphere limiting the degrees of
            freedom on, values must be in [0, 1]. Default is None.
        - radius: Radius of sphere limiting the degrees of freedom. Default is
            None.
        - c_r: Prefactor for covariance matrix biasing.

    Returns:
        Index of last generation. Generation indices start at 1, with 0
            representing the founder.
    """

    founder = pathlib.Path(founder)

    print("\n----------------------------")
    print(
        f"Perform {run_label} for {generations} generations "
        f"starting from founder {founder}."
    )
    print("\n----------------------------")

    dof, transform_dof, dof_atoms = prepare_dof_and_pipeline(
        founder, scaled_center=scaled_center, radius=radius
    )

    # input_pipeline = transform_dof
    thepipeline = InputPipeline(clinamen_pipeline=transform_dof)
    input_pipeline = create_input_pipeline(thepipeline)
    input_pipeline_batch = create_batch_input_pipeline(thepipeline)

    if continue_evolution:
        parameters = continue_with_parameters
        state, population, loss, _, _ = continue_from_checkpoint
    else:
        if scaled_center is not None and c_r is not None:
            initial_cholesky_factor = bias_covariance_matrix(
                atoms=dof_atoms,
                scaled_position=scaled_center,
                c_r=c_r,
                dimension=dof.shape[0],
            )
        else:
            initial_cholesky_factor = None

        parameters, state = cma_setup(
            mean=dof,
            step_size=initial_step_size,
            run_seed=run_seed,
            pop_size=pop_size,
            initial_cholesky_factor=initial_cholesky_factor,
        )
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)

    print(
        f"Starting evolution with dimension {dof.shape[0]} "
        f"and population size {parameters.pop_size} "
        f"and loss_mode '{loss_mode}'."
    )

    evaluate_loss, evaluate_batch_loss = create_mace_energy_single_and_batch(
        models=models
    )

    if loss_mode == ENERGY_ONLY:
        if batch:
            raise NotImplementedError(
                "No batched energy-only evaluation available at the moment."
            )
        else:
            sample_and_evaluate = create_sample_and_sequential_evaluate(
                sample_individuals=sample_individuals,
                evaluate_loss=evaluate_loss,
                input_pipeline=input_pipeline,
            )
    elif loss_mode == LOSS_THRESHOLD:
        if batch:
            single_filter, batch_filter = create_lossthreshold_filter(
                loss_threshold=loss_threshold,
                loss_threshold_exception=LossThresholdException,
            )
            eval_workflow_single = FilterEvalWorkflow(
                filter=single_filter, evaluate_loss=evaluate_loss
            )
            eval_workflow_single = create_filter_eval_workflow(
                eval_workflow_single
            )
            eval_workflow_batch = FilterEvalWorkflow(
                filter=batch_filter, evaluate_loss=evaluate_batch_loss
            )
            eval_workflow_batch = create_filter_eval_workflow(
                eval_workflow_batch
            )
            resample_and_evaluate = create_resample_and_evaluate(
                sample_individuals=sample_individuals,
                evaluate_batch=eval_workflow_batch,
                evaluate_single=eval_workflow_single,
                input_pipeline_batch=input_pipeline_batch,
                input_pipeline_single=input_pipeline,
            )
        else:
            raise NotImplementedError("Loss thresholding requires --batch")
    else:
        raise NotImplementedError(
            f"Loss mode '{loss_mode}' is not available in this version."
        )

    # save evolution
    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )

    termination_criteria = [
        EqualFunValuesCriterion(parameters=parameters, atol=1e-5),
        TolXUpCriterion(parameters=parameters, interpolative=False),
        StaleLossCriterion(
            parameters=parameters, threshold=1e-4, generations=100
        ),
        StaleStepCriterion(
            parameters=parameters, threshold=1e-4, generations=100
        ),
    ]
    termination_criterion = CriteriaOr(
        parameters=parameters, criteria=termination_criteria
    )
    termination_criterion_state = termination_criterion.init()

    handler.save_evolution(
        initial_parameters=parameters,
        initial_state=state,
        additional={
            "model_name": str(models),
            "scaled_center": scaled_center,
            "radius": radius,
            "c_r": c_r,
        },
    )

    if loss_mode == ENERGY_ONLY:
        last_gen = 0
        with tqdm(
            range(generations),
            bar_format="{l_bar}{bar}{r_bar}",
            postfix={"loss": 0.0, "step": 0.0, "std": 0.0},
        ) as t:
            for g in tqdm(range(generations)):
                generation = []
                loss = []
                generation, state, loss = sample_and_evaluate(state=state)
                last_gen = g
                idx = np.argsort(loss)
                state = update_state(state, generation[idx])
                termination_criterion_state = termination_criterion.update(
                    criterion_state=termination_criterion_state,
                    state=state,
                    population=generation,
                    loss=loss,
                )
                terminate = termination_criterion.met(
                    termination_criterion_state
                )
                # save every nth generation and the last one
                if (not (g % save_nth)) or (g == generations - 1) or terminate:
                    handler.save_generation(
                        current_state=state,
                        population=generation,
                        loss=loss,
                        termination_state=termination_criterion_state,
                        json_encoder=JSONEncoder,
                    )
                t.set_postfix(
                    {
                        "loss": loss.min(),
                        "step": state.step_size,
                        "std": loss.std(),
                    }
                )
                t.update(1)
                if terminate:
                    print(
                        f"Termination criterion met after {g+1} generations."
                    )
                    break
    elif loss_mode == LOSS_THRESHOLD:
        last_gen = 0
        with tqdm(
            range(generations),
            bar_format="{l_bar}{bar}{r_bar}",
            postfix={"loss": 0.0, "step": 0.0, "std": 0.0},
        ) as t:
            for g in tqdm(range(generations)):
                generation = []
                loss = []
                (
                    generation,
                    state,
                    loss,
                    information,
                    _,
                ) = resample_and_evaluate(
                    state=state,
                    n_samples=parameters.pop_size,
                    return_failures=True,
                )
                last_gen = g
                idx = np.argsort(loss)
                state = update_state(state, generation[idx])
                termination_criterion_state = termination_criterion.update(
                    criterion_state=termination_criterion_state,
                    state=state,
                    population=generation,
                    loss=loss,
                )
                terminate = termination_criterion.met(
                    termination_criterion_state
                )
                # save every nth generation and the last one
                if (not (g % save_nth)) or (g == generations - 1) or terminate:
                    handler.save_generation(
                        current_state=state,
                        population=generation,
                        loss=loss,
                        termination_state=termination_criterion_state,
                        additional={"information": information},
                        json_encoder=JSONEncoder,
                    )
                t.set_postfix(
                    {
                        "loss": loss.min(),
                        "step": state.step_size,
                        "std": loss.std(),
                    }
                )
                t.update(1)
                if terminate:
                    print(f"Termination criterion met after {g} generations.")
                    break
    else:
        raise NotImplementedError(
            f"Loss mode '{loss_mode}' requires flag '--batch'."
        )

    print(
        f"Minimum loss {loss.min()} for individual "
        f"after {last_gen} generations."
    )

    # write index of last generation to evolution
    handler.update_evolution(
        additional={"last_gen": last_gen + 1, "founder": str(founder)}
    )

    return last_gen, loss.min(), parameters, state
