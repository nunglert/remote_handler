#!/bin/bash
#SBATCH -J AL_boss
#SBATCH --partition=normal
#SBATCH -n 1
#SBATCH -t 0

echo $SLURM_JOB_NODELIST
echo $NODE_SCRATCH_DIR


export REMOTE_HANDLER_CONFIG_DIR=/home/mketter/active_learning_SSPD/config/servers
export REMOTE_HANDLER_SLURM_TEMPLATES=/home/mketter/active_learning_SSPD/config/templates

source /home/mketter/packages/miniconda3/bin/activate
conda activate remote_handler

python run_al.py
