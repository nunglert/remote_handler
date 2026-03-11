eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate JAX-MACE

python GenerateInitialDatabase.py --SEED 66 --N 150 --OUTPUT InitialDatabase.extxyz --VIEW True
