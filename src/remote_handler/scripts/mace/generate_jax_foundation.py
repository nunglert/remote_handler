"""Generate a mace-jax foundation model from a MACE-MP checkpoint.

A one-off bootstrap utility: downloads a MACE-MP foundation checkpoint via
MACE's own API and converts it to the mace-jax ``.pkl`` bundle that jax-based
samplers / finetuning start from -- so you don't have to obtain a torch
``.model`` and convert it by hand.

Run it where the MACE stacks are importable and the download has network
access (typically a cluster login node), e.g.::

    python -m remote_handler.scripts.mace.generate_jax_foundation \\
        --model medium --output foundation-jax.pkl

Needs the ``mace_jax`` extra (torch MACE + mace-jax).
"""

import argparse
from pathlib import Path

from remote_handler.scripts.mace.mace_utils import (
    generate_jax_foundation_model,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="medium",
        help="MACE-MP size (small/medium/large) or a checkpoint path/URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .pkl (default: <checkpoint>-jax.pkl).",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Optional head name to record in the bundle.",
    )
    args = parser.parse_args(argv)

    out = generate_jax_foundation_model(
        model=args.model, output_path=args.output, head=args.head
    )
    print(f"Wrote mace-jax foundation model to: {out}")


if __name__ == "__main__":
    main()
