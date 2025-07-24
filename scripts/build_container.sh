# !/bin/bash
#
# Build script for Perovskite GCNN Singularity Container
# This script builds the container for HPC environments

#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --mem=64000
#SBATCH --time=02:00:00


echo "Building Perovskite GCNN Singularity Container..."

# Check if Singularity/Apptainer is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Error: Neither apptainer nor singularity found. Please install one of them."
    exit 1
fi

echo "Using container command: $CONTAINER_CMD"

# Set container name
CONTAINER_NAME="perovskite_exp.sif"

# Build the container
echo "Building container: $CONTAINER_NAME"
echo "This may take 30-60 minutes depending on your internet connection and system..."

$CONTAINER_CMD build --fakeroot $CONTAINER_NAME perovskite_exp.def

# Test the container
echo "Testing the container..."
$CONTAINER_CMD test $CONTAINER_NAME

echo ""
echo "Container build completed successfully!"
echo "Container file: $CONTAINER_NAME"
echo ""
echo "Usage examples:"
echo "  Interactive shell: $CONTAINER_CMD shell $CONTAINER_NAME"
echo "  Run Python script: $CONTAINER_CMD exec $CONTAINER_NAME python script.py"
echo "  Run with GPU: $CONTAINER_CMD exec --nv $CONTAINER_NAME python script.py"
echo "  Jupyter notebook: $CONTAINER_CMD exec --nv $CONTAINER_NAME jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "For HPC usage, copy the .sif file to your HPC system and use with SLURM jobs." 
