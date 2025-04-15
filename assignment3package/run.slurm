#!/bin/bash
#SBATCH --job-name=assig3
#SBATCH --output=out_%j.log    
#SBATCH --error=err_%j.log
#SBATCH --ntasks=16                          # Number of tasks (processes)
#SBATCH --time=24:00:00                     # Time limit (hh:mm:ss)
#SBATCH --mem=20G                            # Memory per node
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL           # When to send email (BEGIN, END, FAIL, or ALL)
#SBATCH --mail-user=u1527533@utah.edu              # Replace with your email address
#SBATCH --account=tolgalab
#SBATCH --partition=titanrtx-24              # titanrtx-24 , a6000
#SBATCH --nodelist=eris 

# Load modules (if needed)
# module load cuda/12.6.0

# Run the Python script
python -u assignment3.py