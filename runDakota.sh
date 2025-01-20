#!/bin/bash
#SBATCH --job-name="dakota"
#SBATCH --time=2:00:00
#SBATCH --mem=0
dakota dakota_vagus.in

