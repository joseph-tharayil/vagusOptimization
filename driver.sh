#!/bin/bash
#SBATCH --job-name="dakota_driver"
#SBATCH --time=2:00:00
#SBATCH --mem=0

sbatch driver_real.sh $1 $2 > sbatch.out
jobid=$(tail -1 sbatch.out | egrep -o '[0-9]+')
while [ $(squeue -j $jobid | wc -l) -ne 0 ];
do
sleep 300
done
