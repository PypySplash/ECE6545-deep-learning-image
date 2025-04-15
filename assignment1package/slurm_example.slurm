#!/bin/bash
#SBATCH --job-name=assig1_job               # Job name
#SBATCH --output=testoutput_%j.log          # Output log file (with job ID)
#SBATCH --error=testerror_%j.log            # Error log file (with job ID)
#SBATCH --ntasks=2                          # Number of tasks (processes)
#SBATCH --time=01:00:00                     # Time limit (hh:mm:ss)
#SBATCH --mem=4G                            # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL          # When to send email (BEGIN, END, FAIL, or ALL)
#SBATCH --mail-user=id@utah.edu             # Replace with your email address

# Load modules (if needed)
module load python/3.6.8                    # Example: Load Python module

# Activate your virtual environment (if you have one)
source ~/venvs/my_env/bin/activate

# Install required Python libraries
pip install --user numpy                    # Replace 'numpy' with the library you need

# Run the Python script
python assignment1_ex1.py