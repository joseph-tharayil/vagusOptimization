#!/bin/bash
#SBATCH --job-name="dakota"
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH -C clx
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --account=proj85
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
##SBATCH --array=0-5

module purge
module load unstable python-dev
source ~/conntilitEnv/bin/activate

srun -n 1 python combineSignals.py #$SLURM_ARRAY_TASK_ID
