#!/bin/bash
#SBATCH --job-name="dakota_driver"


source ../envDakota/bin/activate
python driver.py $1 $2

