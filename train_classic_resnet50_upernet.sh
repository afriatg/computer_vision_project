#!/bin/bash

#SBATCH --job-name=CV
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00
#SBATCH --mem-per-cpu=32G

# Module load
module load python/3.9.4

# Train the networks
xvfb-run -d python3 /home/afriatg/projects/computer_vision_project/train.py --use_disc 0