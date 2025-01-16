#!/bin/bash
#SBATCH --job-name="dakota"
#SBATCH --partition=prod
#SBATCH --nodes=18
#SBATCH -C clx
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --account=proj85
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-5

module purge
module load unstable python-dev
source ~/conntilitEnv/bin/activate

filename='/gpfs/bbp.cscs.ch/project/proj85/scratch/vagusNerve/optimization/signals'

echo $SLURM_ARRAY_TASK_ID

if [ $SLURM_ARRAY_TASK_ID -eq 2 ] 

    then

    echo $SLURM_ARRAY_TASK_ID

    rm -r $filename
    mkdir $filename



    mkdir $filename/maff

    for i in {0..5}
    do
        for j in {0..215}
        do
            folder="$filename/maff/$i/$j"
                mkdir -p $folder 2>/dev/null

                done
        done



    for i in {0..5}
    do
        for j in {0..215}
        do
            folder="$filename/meff/$i/$j"
                mkdir -p $folder 2>/dev/null

                done

        done
    fi

    wait


srun -n 39 python analytic.py $SLURM_ARRAY_TASK_ID
