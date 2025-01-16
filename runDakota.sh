#!/bin/bash
#SBATCH --job-name="dakota"
#SBATCH --partition=prod
#SBATCH --nodes=39
#SBATCH -C clx
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --account=proj85
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0

source ~/dakotaEnv/bin/activate
dakota dakota_vagus.in

