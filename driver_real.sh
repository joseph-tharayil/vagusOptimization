#!/bin/bash
#SBATCH --job-name="dakota_driver"
#SBATCH --time=2:00:00
#SBATCH --mem=0

source ~/dakotaEnv/bin/activate
python driver.py $1 $2

