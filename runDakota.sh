#!/bin/bash
#SBATCH --job-name="dakota"
#SBATCH --time=2:00:00

dakota dakota_vagus.in

