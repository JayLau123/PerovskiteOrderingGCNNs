#!/bin/bash
#
#SBATCH --job-name=perovskite_exp   # Job name
#SBATCH --output=logs/exp_%A_%a.log # Log file for each job, %A=job ID, %a=task ID
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=16           # Number of CPU cores per task
#SBATCH --mem=64G                  # Memory per job
#SBATCH --gres=gpu:1                # Request 1 GPU per job
#SBATCH --time=72:00:00             # Time limit for each job
#
# --- HPC Cluster Specific Settings (IMPORTANT) ---
# Set the partition and QoS for the job.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#
#SBATCH --array=0-3                 # JOB ARRAY: 4 jobs for the 4 configurations

# --- Navigate to Project Root ---
# This is the critical fix. SLURM_SUBMIT_DIR is an environment variable
# that holds the directory where the 'sbatch' command was executed.
# This ensures the job runs in your project's main directory.
cd "$SLURM_SUBMIT_DIR"

# --- Create Log Directory ---
# Create a directory for log files if it doesn't exist
mkdir -p logs

# --- Define Experiment Parameters ---
# Create arrays of the parameters you want to test
models=("CGCNN" "e3nn")
datasets=("relaxed" "unrelaxed")

# --- Map Array Task ID to Parameters ---
# This logic maps the SLURM_ARRAY_TASK_ID (0-3) to a unique configuration
# Model index (0 or 1): Changes every 2 jobs
model_index=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]}))
MODEL=${models[$model_index]}

# Dataset index (0 or 1): Alternates for each job
dataset_index=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))
DATASET=${datasets[$dataset_index]}

# --- Print Job Configuration to Log ---
echo "----------------------------------------------------"
echo "Starting SLURM Job Array: Job ID $SLURM_JOB_ID, Task ID $SLURM_ARRAY_TASK_ID"
echo "Working Directory: $(pwd)"
echo "Model Type: $MODEL"
echo "Structure Type (Dataset): $DATASET"
echo "Container: perovskite_v100.sif"
echo "----------------------------------------------------"

# --- Execute the Experiment ---
# Updated to use the new apptainer command format with environment activation and cache setup
apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace perovskite_exp.sif bash -c "
source activate perovskite_env
# Create cache directories in workspace
mkdir -p /workspace/cache/{matplotlib,pip,fontconfig}
# Redirect all cache directories to workspace
export MPLCONFIGDIR=/workspace/cache/matplotlib
export PIP_CACHE_DIR=/workspace/cache/pip
export FONTCONFIG_PATH=/workspace/cache/fontconfig
export XDG_CACHE_HOME=/workspace/cache
export PYTHONPATH=/workspace:/workspace/local_packages:\$PYTHONPATH
python training/run_wandb_experiment.py --struct_type $DATASET --model $MODEL --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
"

echo "Job finished successfully."