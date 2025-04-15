#!/bin/bash
#SBATCH --job-name=dl_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=kingspeak-gpu
#SBATCH --account=kingspeak-gpu
#SBATCH --mail-type=BEGIN,END,FAIL          # When to send email (BEGIN, END, FAIL, or ALL)
#SBATCH --mail-user=id@utah.edu

echo "Current directory: $(pwd)"
echo "Current date and time: $(date)"
# nvidia-smi

# Run the Python script
echo "*** starting ex1 ***"
python assignment2_ex1.py
echo "*** finished ex1 ***"

echo "*** starting ex2 ***"
python assignment2_ex2.py
echo "*** finished ex2 ***"