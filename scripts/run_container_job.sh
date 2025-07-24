#!/bin/bash
#
#SBATCH --job-name=perovskite_exp_single   # Job name
#SBATCH --output=logs/exp_single_%j.log    # Log file, %j=job ID
#SBATCH --ntasks=1                         # Run on a single CPU
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --mem=64G                          # Memory per job
#SBATCH --gres=gpu:1                       # Request 1 GPU per job
#SBATCH --time=72:00:00                    # Time limit for the job
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

# --- Navigate to Project Root ---
cd "$SLURM_SUBMIT_DIR"

# --- Create Log Directory ---
mkdir -p logs

# --- Set Experiment Parameters ---
MODEL="CGCNN"         # or "e3nn"
DATASET="relaxed"     # or "unrelaxed"

echo "----------------------------------------------------"
echo "Starting SLURM Job: Job ID $SLURM_JOB_ID"
echo "Working Directory: $(pwd)"
echo "Model Type: $MODEL"
echo "Structure Type (Dataset): $DATASET"
echo "Container: perovskite_exp.sif"
echo "----------------------------------------------------"

# --- Execute the Experiment ---
apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace perovskite_exp.sif bash -c "
source activate perovskite_env
mkdir -p /workspace/cache/{matplotlib,pip,fontconfig}
export MPLCONFIGDIR=/workspace/cache/matplotlib
export PIP_CACHE_DIR=/workspace/cache/pip
export FONTCONFIG_PATH=/workspace/cache/fontconfig
export XDG_CACHE_HOME=/workspace/cache
export PYTHONPATH=/workspace:/workspace/local_packages:\$PYTHONPATH
# Optionally, to resume a previous sweep, add --resume_sweep_id <sweep_id> to the command below
# Example:
# python training/run_wandb_experiment.py --struct_type $DATASET --model $MODEL --gpu 0 --budget 50 --training_fraction 1 --training_seed 0 --resume_sweep_id <sweep_id>
python training/run_wandb_experiment.py --struct_type $DATASET --model $MODEL --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
"

echo "Job finished successfully."