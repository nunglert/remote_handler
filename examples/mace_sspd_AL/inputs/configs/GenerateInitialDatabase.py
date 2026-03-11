import sys
import os

import argparse
import pickle
import ase.io
import random
import numpy as np

from ase.data.pubchem import pubchem_atoms_search
from ase.visualize import view



parser = argparse.ArgumentParser(
    description="Get parameters for the initial database creation"
)


parser.add_argument(
    "--N",
    required=True,
    type=int,
    help="Number of configs to create",
    dest="N",
)
parser.add_argument(
    "--OUTPUT",
    required=True,
    type=str,
    help="Folder where the database are stored",
    dest="OUTPUT",
)
parser.add_argument(
    "--SEED",
    required=True,
    type=int,
    help="Initial seed for the random module",
    dest="SEED",
)
parser.add_argument(
    "--VIEW",
    required=True,
    type=bool,
    help="Should the created database be visualized using ase.view",
    dest="VIEW",
)

args = parser.parse_args()

random.seed(args.SEED)
ground_truth_struct = pubchem_atoms_search("isopropanol")

structs = []
for _ in range(args.N):
    struct = ground_truth_struct.copy()
    seed = random.randint(1, 5000)
    stdev = random.uniform(0.025, 0.1)
    while True:
        struct.rattle(stdev=stdev, seed=seed)
        dists = struct.get_all_distances()
        indizes = np.tril_indices_from(dists, k=-1)
        min_dist = np.min(dists[indizes])
        if min_dist >= 0.5:
            break
    structs.append(struct)
print("Generated database of length: " + str(len(structs)))

if args.VIEW:
    view(structs)

ase.io.write(args.OUTPUT, structs)

