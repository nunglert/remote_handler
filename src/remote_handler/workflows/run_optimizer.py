import argparse
import ast
import json
import logging
import os
import re
from pathlib import Path

from remote_handler.core import file_handler, file_utils, utils
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows import optimizer

# from jaxnest import file_utils
# from jaxnest.file_utils import grep, secure_path
# from jaxnest.postprocess.optimizer import RemoteOptimizationHandler, SimpleLogger
# from jaxnest.postprocess import optimizer

logger = logging.getLogger(__name__)

"""
Usage:

    Without a config file:

        python /home/nunglert/code/jaxns/scripts/utility/run_optimizer.py \
            --path output_data \
            --local_machine lugus \
            --remote_machine lugus \
            --python_script_path /home/nunglert/code/jaxns/scripts/utility/optimize_traj.py \
            --do_steinhardt \
            --do_optimization \
            --optimization_backend LJ \
            --do_traj \
            --every 10 \
            --shard_size 50 \
            --rmax 3.0 \
            --partition normal \
            --num_parallel 10 \
            --jobname opt


    Using a config file in json format that contains parameters:

        python /home/nico.unglert/code/jaxnest_dev/scripts/utility/run_optimizer.py \
            --config /share/nico.unglert/experiments/silicon/xrun_opts/run_trajs/run_stein.json \
            --path output_data \
            --do_traj \
            --num_parallel 50 

    
    Rerunning a crashed optimizer run:

        python /home/nunglert/code/jaxns/scripts/utility/run_optimizer.py \
            --recompute_failed \
            --read_out_auto \
            --partition paral


"""


def get_pretty_args(args):
    """
    Pretty-print the arguments parsed by argparse.
    """
    return json.dumps(vars(args), indent=4, sort_keys=True)


def load_args_from_json(json_file):
    """
    Load arguments from a JSON file and return as a dictionary.
    """
    with open(json_file, "r") as f:
        return json.load(f)


def merge_args_with_json(args, json_args, parser):
    """
    Merge JSON arguments with command-line arguments.
    JSON arguments will override values in args with the ones from
    json_args if the value in args is not equal to the default
    value.

    WARNING:
    Should find a more stable way to prefer command line arguments
    over config file arguments. At the moment this works as
    intended, as long as the argument passend in the command line
    does not equal the default value.
    """
    for key, value in json_args.items():
        if getattr(args, key, None) == parser.get_default(key):
            setattr(args, key, value)
    return args


def parse_json_dict(log_text):
    """
    Parses the JSON dictionary part from the provided text and returns it as a Python dictionary.

    Args:
        log_text (str): The text containing the JSON dictionary.

    Returns:
        dict: The parsed JSON dictionary.
    """
    # Regular expression to extract the JSON dictionary block
    json_block_pattern = re.compile(r"Parsed arguments:\s*({.*?})", re.DOTALL)
    match = json_block_pattern.search(log_text)

    if not match:
        raise ValueError("No JSON dictionary found in the provided text.")

    json_block = match.group(1)  # Extract the JSON block

    # Parse the JSON block into a Python dictionary
    parsed_dict = json.loads(json_block)
    return parsed_dict


def parse_write_path(log_text):
    """
    Parses the path after 'Writing files to' in the provided log text.

    Args:
        log_text (str): The text to parse.

    Returns:
        str: The extracted path.

    Raises:
        ValueError: If no matching path is found.
    """
    # Regular expression to match the path after 'Writing files to'
    # match = re.search(r"Writing files to\s+(\S+)", log_text)
    match = re.search(r"rootpath:\s+(\S+)", log_text)
    if match:
        return match.group(1)  # Extract the matched path
    else:
        raise ValueError("No path found after 'Writing files to'.")


def find_newest_log(directory, pre_tag="opt", post_tag=r"\.log"):
    # Define a regular expression to match relevant files
    log_pattern = re.compile(
        r"^" + pre_tag + r"(_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})?" + post_tag
    )

    # Filter files that match the pattern
    relevant_files = [f for f in os.listdir(directory) if log_pattern.match(f)]
    # print(relevant_files)

    # Get the full path and modification time for each relevant file
    files_with_mtime = [
        (f, os.path.getmtime(os.path.join(directory, f)))
        for f in relevant_files
    ]

    # Sort files by modification time (most recent first)
    newest_file = (
        max(files_with_mtime, key=lambda x: x[1])[0]
        if files_with_mtime
        else None
    )

    return newest_file


def get_parser():
    parser = argparse.ArgumentParser(
        description="Parser for optimization script parameters."
    )

    parser.add_argument(
        "--shard_size", type=int, default=1000, help="Size of each shard."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        help="Path to save the results.",
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=100,
        help="Number of parallel tasks.",
    )
    parser.add_argument(
        "--n_processors",
        type=int,
        default=1,
        help="Number of processors per task.",
    )
    parser.add_argument(
        "--python_script_path",
        type=str,
        default=None,
        help="Remote payload to run. Defaults to the shipped script for the "
        "selected mode (optimize_traj.py / analyze_traj.py).",
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        help="Path to the pickle file with NNFF/MACE model parameters "
        "(required for those optimization backends).",
    )
    parser.add_argument(
        "--supercell_trafo",
        type=str,
        default="[3, 3, 3]",
        help="Transformation for the supercell as a list of integers (e.g., '[3, 3, 3]').",
    )
    parser.add_argument(
        "--max_neighbors_list",
        type=str,
        default="[15, 20, 25, 30, 35, 40, 45, 50, 60]",
        help="List of maximum neighbors as a string (e.g., '[15, 20, 25]').",
    )
    parser.add_argument(
        "--max_neighbors_offset",
        type=int,
        default=3,
        help="Offset for max neighbors.",
    )
    parser.add_argument(
        "--optimization_backend",
        type=str,
        default="NNFF",
        help="Backend for optimization. Also supports ASE LJ.",
    )
    parser.add_argument(
        "--steinhardt_l_values",
        type=str,
        default="[4, 6]",
        help="L values for which to compute Steinhardt Q and W",
    )
    parser.add_argument(
        "--loop_time",
        type=int,
        default=5,
        help="Time for each loop iteration in seconds.",
    )
    parser.add_argument(
        "--jobname", type=str, default="opt", help="Name of the job."
    )
    parser.add_argument(
        "--stub", action="store_true", help="Use stub mode (default: False)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of steps for optimization.",
    )
    parser.add_argument(
        "--fmax", type=float, default=0.1, help="Force convergence threshold."
    )
    parser.add_argument(
        "--rmax", type=float, default=4.0, help="Maximum cutoff radius."
    )
    parser.add_argument(
        "--nbins", type=int, default=200, help="Number of bins for RDF."
    )
    parser.add_argument(
        "--every", type=int, default=1, help="Frequency of saving results."
    )
    parser.add_argument(
        "--do_steinhardt",
        action="store_true",
        help="Enable Steinhardt parameter calculation.",
    )
    parser.add_argument(
        "--do_optimization", action="store_true", help="Enable optimization."
    )
    parser.add_argument(
        "--do_rdf", action="store_true", help="Enable RDF calculation."
    )
    parser.add_argument(
        "--do_traj",
        action="store_true",
        default=False,
        help="Search for traj files in path and process them",
    )
    parser.add_argument(
        "--do_walkers",
        action="store_true",
        default=False,
        help="Search for walkers files in path and process them",
    )
    parser.add_argument(
        "--custom_regex",
        type=str,
        help="Search for walkers files in path and process them",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Specific extxyz file(s) to process. Provide paths without separator",
    )
    parser.add_argument(
        "--log_mode",
        type=str,
        default="stdout",
        help="Log output to console (stdout) or to a file (file). If --logfile is not specified, we write the logfile either to the parent folder of --path, if specified. Otherwise, to the current working directory.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="File to print output",
    )
    parser.add_argument(
        "--partition",
        default="normal",
        type=str,
        help="File to print output",
    )
    parser.add_argument(
        "--config", type=str, help="Path to a JSON configuration file."
    )
    parser.add_argument(
        "--path", type=str, help="Path to look for certain file types."
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        default=False,
        help="Don't run processing, only try to collect results in savepath",
    )
    parser.add_argument(
        "--recompute_failed",
        action="store_true",
        default=False,
        help="Check for failed shards and try to recompute them -> collect all",
    )
    parser.add_argument(
        "--read_out",
        default=None,
        type=str,
        help="Parse a log file for configuration",
    )
    parser.add_argument(
        "--remote_machine",
        default="datalab",
        type=str,
        help="Name of remote machine",
    )
    parser.add_argument(
        "--local_machine",
        default="dirona",
        type=str,
        help="Name of local machine",
    )
    parser.add_argument(
        "--read_out_auto",
        action="store_true",
        default=False,
        help="Automatically find newest log file in cwd and parse it for configuration",
    )
    return parser


def process_args(args, parser):
    if args.path is not None:
        args.path = str(Path(args.path).absolute())

    if args.config:
        # Load JSON config if provided
        json_args = load_args_from_json(args.config)
        args = merge_args_with_json(args, json_args, parser)

    elif args.read_out or args.read_out_auto:
        # Read args from a logfile if provided

        if args.read_out_auto:
            args.read_out = find_newest_log(os.getcwd())

        with open(args.read_out, "r") as f:
            content = f.read()
        args_dict = parse_json_dict(content)
        args = merge_args_with_json(args, args_dict, parser)

        if args.recompute_failed:
            args.collect_only = True

        if args.collect_only:
            args.savepath = parse_write_path(content)

    # Convert stringified lists to Python lists
    if isinstance(args.supercell_trafo, str):
        args.supercell_trafo = ast.literal_eval(args.supercell_trafo)
    if isinstance(args.max_neighbors_list, str):
        args.max_neighbors_list = ast.literal_eval(args.max_neighbors_list)
    if isinstance(args.steinhardt_l_values, str):
        args.steinhardt_l_values = ast.literal_eval(args.steinhardt_l_values)

    if args.log_mode == "file":
        if args.logfile is None:
            if args.path is not None:
                homepath = Path(args.path).parent
            else:
                homepath = Path(os.getcwd())
            args.logfile = str(file_utils.secure_path(homepath / "opt.log"))
        logfile = args.logfile

    elif args.log_mode == "stdout":
        logfile = None

    opt_logger = utils.SimpleLogger(logfile)

    opt_logger.print("\nParsed arguments:")
    opt_logger.print(get_pretty_args(args))

    args.ats_files = []
    if args.path:
        if args.do_traj:
            traj_files = utils.grep(
                args.path, r"^ns\.\d+\.traj.extxyz", with_path=True
            )
            args.ats_files.extend(traj_files)
            logger.info(f"Found {len(traj_files)} trajectory files.")

        if args.do_walkers:
            walkers_files = utils.grep(
                args.path, r"^ns\.\d+\.full_snap\.\d+\.extxyz", with_path=True
            )
            args.ats_files.extend(walkers_files)
            logger.info(f"Found {len(walkers_files)} walkers files.")

        if args.custom_regex is not None:
            atoms_files = utils.grep(
                args.path, rf"{args.custom_regex}", with_path=True
            )
            args.ats_files.extend(atoms_files)
            logger.info(f"Found {len(atoms_files)} files with custom regex.")

    if args.files:
        args.ats_files.extend(args.files)

    if not len(args.ats_files) > 0:
        raise Exception(
            "Need some extxyz files to process! Specify either --files "
            "directly, or provide a --path and set --do_walkers or --do_traj "
            "to automatically look for extxyz files in the path."
        )

    opt_logger.print("\nParsed extxyz files:")
    for file in args.ats_files:
        opt_logger.print(file)

    if args.remote_machine == "local":
        args.remote_fs = file_handler.LocalFileSystem()
        # If no savepath provided, use temp directory
        if not args.savepath:
            args.savepath = str(args.remote_fs.create_tempdir())
        # Create a minimal local cluster config
        args.cluster_config = ClusterConfig(
            remote_machine="local",
            partition=args.partition,
            conda_env="",
            conda_source="",
            savepath=args.savepath,
            template="local.sh.j2",
        )
    else:
        args.remote_fs = file_handler.RemoteFileSystem(args.remote_machine)
        # If no savepath provided, use temp directory on remote
        if not args.savepath:
            args.savepath = str(args.remote_fs.create_tempdir())
        # Create a cluster config for remote machine
        args.cluster_config = ClusterConfig(
            remote_machine=args.remote_machine,
            partition=args.partition,
            conda_env="jaxnest",  # Default conda environment
            conda_source="~/miniconda3/etc/profile.d/conda.sh",  # Default conda source
            savepath=args.savepath,
            template=f"{args.remote_machine}.sh.j2",
        )

    return args


def _shared_handler_kwargs(args):
    """Sharding / scheduling kwargs common to both handler types."""
    return dict(
        cluster_config=args.cluster_config,
        remote_fs=args.remote_fs,
        n_processors=args.n_processors,
        shard_size=args.shard_size,
        savepath=args.savepath,
        num_parallel=args.num_parallel,
        python_script_path=args.python_script_path,
        loop_time=args.loop_time,
        jobname=args.jobname,
        stub=args.stub,
        every=args.every,
        logfile=args.logfile,
        collect_only=args.collect_only,
    )


def setup_optimizer(args):
    """Build a :class:`~remote_handler.workflows.optimizer.RemoteOptimizationHandler`."""
    return optimizer.RemoteOptimizationHandler(
        args.ats_files,
        pickle_file=args.pickle_file,
        optimization_backend=args.optimization_backend,
        supercell_trafo=args.supercell_trafo,
        max_neighbors_list=args.max_neighbors_list,
        max_neighbors_offset=args.max_neighbors_offset,
        steps=args.steps,
        fmax=args.fmax,
        **_shared_handler_kwargs(args),
    )


def setup_analysis(args):
    """Build a :class:`~remote_handler.workflows.optimizer.RemoteAnalysisHandler`."""
    l_values = args.steinhardt_l_values
    return optimizer.RemoteAnalysisHandler(
        args.ats_files,
        do_steinhardt=args.do_steinhardt,
        do_rdf=args.do_rdf,
        steinhardt_l_value=(
            l_values[0] if isinstance(l_values, list) else l_values
        ),
        cutoff=args.rmax,
        nbins=args.nbins,
        **_shared_handler_kwargs(args),
    )


def setup_handler(args):
    """Pick the handler for the requested mode.

    Optimization and analysis are now separate concerns -- one invocation does
    one of them. Optimization takes precedence if both are somehow requested.
    """
    if args.do_optimization:
        return setup_optimizer(args)
    if args.do_steinhardt or args.do_rdf:
        return setup_analysis(args)
    raise ValueError(
        "Nothing to do: enable --do_optimization, or --do_steinhardt / "
        "--do_rdf for analysis."
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args = process_args(args, parser)

    handler = setup_handler(args)

    if args.recompute_failed:
        handler.recompute_failed()
    elif args.collect_only:
        handler.collect_and_harvest()
    else:
        handler.run()
