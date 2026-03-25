"""Convert an h5md trajectory file to extxyz format.

Usage:
    python -m remote_handler.scripts.torch_sim.h5md_to_extxyz trajectory.h5md output.extxyz
    python -m remote_handler.scripts.torch_sim.h5md_to_extxyz trajectory.h5md  # writes to trajectory.extxyz
"""

import argparse
from pathlib import Path

import ase.io
from torch_sim.trajectory import TorchSimTrajectory


def convert_h5md_to_extxyz(h5md_path: Path, output_path: Path) -> None:
    """Read an h5md trajectory and write all frames as extxyz.

    Args:
        h5md_path: Path to the input .h5md file.
        output_path: Path to the output .extxyz file.
    """
    traj = TorchSimTrajectory(str(h5md_path), mode="r")
    frames = traj.get_atoms()
    traj.close()

    if not isinstance(frames, list):
        frames = [frames]

    ase.io.write(str(output_path), frames)
    print(f"Wrote {len(frames)} frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an h5md trajectory to extxyz format."
    )
    parser.add_argument(
        "h5md_file", type=Path, help="Input .h5md trajectory file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=None,
        help="Output .extxyz file (default: <input_stem>.extxyz)",
    )
    args = parser.parse_args()

    if not args.h5md_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.h5md_file}")

    output = args.output_file or args.h5md_file.with_suffix(".extxyz")
    convert_h5md_to_extxyz(args.h5md_file, output)


if __name__ == "__main__":
    main()
