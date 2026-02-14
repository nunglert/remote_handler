#!/bin/bash
#SBATCH --job-name=cmaes_sample
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00


source /home/nico.unglert/miniconda3/bin/activate
conda activate clinamen

python /home/nico.unglert/dump/remote_handler/cmaes_sample_2026-02-14_15_06_17YUuljTxsIS/cmaes_sampling.py /home/nico.unglert/dump/remote_handler/cmaes_sample_2026-02-14_15_06_17YUuljTxsIS/input_cmaes.json
