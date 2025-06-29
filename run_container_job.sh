#!/bin/bash
#SBATCH --job-name=perovskite_gcnn
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=perovskite_gcnn_%j.out
#SBATCH --error=perovskite_gcnn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@institution.edu

# Load any required modules (modify as needed for your HPC system)
# module load cuda/11.7
# module load singularity/3.9.0

# Set container cache directory to job scratch directory for faster builds
export APPTAINER_CACHEDIR=$SLURMTMPDIR

# Set working directory
cd $SLURM_SUBMIT_DIR

# Create output directory
mkdir -p outputs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check if container exists, if not pull it
if [ ! -f "perovskite_gcnn.sif" ]; then
    echo "Container not found, pulling from Docker Hub..."
    apptainer pull perovskite_gcnn.sif docker://nvcr.io/nvidia/pytorch:25.01-py3
    echo "Container pulled successfully"
fi

# Test GPU access
echo "Testing GPU access..."
apptainer exec --nv perovskite_gcnn.sif nvidia-smi

# Run your Python script (modify the script name as needed)
echo "Running Perovskite GCNN training..."
apptainer exec --nv \
    --bind $SLURM_SUBMIT_DIR:/workspace \
    --bind $SLURM_SUBMIT_DIR/data:/workspace/data \
    --bind $SLURM_SUBMIT_DIR/outputs:/workspace/outputs \
    perovskite_gcnn.sif \
    python /workspace/1_model_training.ipynb

# Alternative: Run a Python script directly
# apptainer exec --nv \
#     --bind $SLURM_SUBMIT_DIR:/workspace \
#     --bind $SLURM_SUBMIT_DIR/data:/workspace/data \
#     --bind $SLURM_SUBMIT_DIR/outputs:/workspace/outputs \
#     perovskite_gcnn.sif \
#     python /workspace/your_script.py

echo "Job completed at $(date)" 