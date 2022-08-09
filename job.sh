#!/bin/bash


#SBATCH --job-name=job_name
#SBATCH --partition=talon-gpu
#SBATCH --export=ALL
#SBATCH --time=28-00:00:00
#SBATCH --nodes=1
#SBATCH --error=./err_%x_%j.txt
#SBATCH --output=./out_%x_%j.txt
#SBATCH --mail-type=NONE

module load cudnn8.1-cuda11.2/8.1.1.33
module load cuda10.2/toolkit/10.2.89


python3 --version
python3 program_name.py path_to_training path_to_valid


