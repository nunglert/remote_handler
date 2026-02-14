#!/bin/sh
#SBATCH --job-name=dft
#SBATCH --ntasks=8 
#SBATCH --partition=normal
#SBATCH --time=0-02:00:00
module purge
module load vasp/6.4.2_intel
source /opt/intel/oneapi/setvars.sh --force
export VASP_PP_PATH=/home/nunglert/cellar/vasp/paw

ulimit -u 40000
ulimit -s unlimited

pwd; hostname; date

source /home/nunglert/miniconda3/bin/activate
conda activate jaxnest

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID

MAP=(0)

# Get the mapped index for this SLURM array task
MAPPED_INDEX=${MAP[$SLURM_ARRAY_TASK_ID]}

PER_TASK=1

# Calculate the starting and ending values for this task
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  run_id=$((run - 1))
  python /home/nunglert/dump/dft/dft/dft_singlepoint.py /home/nunglert/dump/dft/dft/calcdirs/$run_id
done
