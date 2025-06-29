# Perovskite GCNN Singularity Container

This directory contains files to create and run a Singularity/Apptainer container for the Perovskite Ordering GCNNs project on HPC systems.

## Files

- `perovskite_gcnn.def` - Singularity container definition file
- `build_container.sh` - Script to build the container
- `run_container_job.sh` - SLURM job script template for HPC execution
- `CONTAINER_README.md` - This file

## Container Contents

The container includes all dependencies from the `environment.yml` file:

- **PyTorch 1.13.1** with CUDA 11.7 support
- **PyTorch Geometric (PyG) 2.3.0** for graph neural networks
- **Pymatgen 2023.5.10** for materials science
- **E3NN 0.5.1** for equivariant neural networks
- **RDKit 2023.3.1** for cheminformatics
- **ASE 3.22.1** for atomic simulation environment
- **Standard ML libraries**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **Wandb 0.16.3** for experiment tracking
- **Jupyter** for interactive development
- **Additional packages**: plotly, networkx, tqdm, and many more

## Building the Container

### Option 1: Using the build script (recommended)

```bash
chmod +x build_container.sh
./build_container.sh
```

### Option 2: Manual build

```bash
# For systems with Apptainer
apptainer build --fakeroot perovskite_gcnn.sif perovskite_gcnn.def

# For systems with Singularity
singularity build --fakeroot perovskite_gcnn.sif perovskite_gcnn.def
```

**Note**: Building may take 30-60 minutes depending on your internet connection and system.

## Testing the Container

After building, test that all packages are available:

```bash
apptainer test perovskite_gcnn.sif
```

## Usage on HPC Systems

### 1. Interactive Session

Request an interactive compute node:

```bash
salloc --partition=general-compute --qos=general-compute --mem=64G --time=2:00:00 --gpus-per-node=1 --ntasks-per-node=8
```

Once on the compute node:

```bash
# Set cache directory for faster container operations
export APPTAINER_CACHEDIR=$SLURMTMPDIR

# Interactive shell
apptainer shell --nv perovskite_gcnn.sif

# Or run a specific command
apptainer exec --nv perovskite_gcnn.sif python your_script.py
```

### 2. Batch Job Submission

Modify the `run_container_job.sh` script with your email and specific requirements, then submit:

```bash
sbatch run_container_job.sh
```

### 3. Jupyter Notebook

For interactive Jupyter notebooks:

```bash
# Start Jupyter server
apptainer exec --nv \
    --bind $PWD:/workspace \
    perovskite_gcnn.sif \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Then SSH tunnel to access the notebook
ssh -L 8888:localhost:8888 username@hpc-cluster
```

## Container Commands

### Basic Usage

```bash
# Interactive shell
apptainer shell --nv perovskite_gcnn.sif

# Execute a command
apptainer exec --nv perovskite_gcnn.sif python script.py

# Run with bind mounts for data access
apptainer exec --nv \
    --bind /path/to/data:/workspace/data \
    --bind /path/to/outputs:/workspace/outputs \
    perovskite_gcnn.sif \
    python script.py
```

### GPU Testing

```bash
# Test GPU access
apptainer exec --nv perovskite_gcnn.sif nvidia-smi

# Test PyTorch GPU
apptainer exec --nv perovskite_gcnn.sif python -c "import torch; print(torch.cuda.is_available())"
```

## HPC-Specific Considerations

### 1. Cache Directory

Set the Apptainer cache to your job's scratch directory for faster operations:

```bash
export APPTAINER_CACHEDIR=$SLURMTMPDIR
```

### 2. Resource Optimization

For faster container builds, request more CPUs:

```bash
salloc --partition=debug --qos=debug --time=1:00:00 --ntasks-per-node=10
```

### 3. Data Binding

Bind your data directories to the container:

```bash
apptainer exec --nv \
    --bind $SLURM_SUBMIT_DIR:/workspace \
    --bind /path/to/data:/workspace/data \
    perovskite_gcnn.sif \
    python script.py
```

### 4. Environment Variables

Set appropriate environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

## Troubleshooting

### Common Issues

1. **Container build fails**: Ensure you have sufficient disk space and internet connection
2. **GPU not detected**: Use the `--nv` flag and ensure CUDA modules are loaded
3. **Permission errors**: Use `--fakeroot` during build or run with appropriate permissions
4. **Package import errors**: Check the container test output for missing packages

### Debugging

```bash
# Check container contents
apptainer shell perovskite_gcnn.sif
ls -la /opt/conda/lib/python3.10/site-packages/

# Test specific package
apptainer exec perovskite_gcnn.sif python -c "import package_name; print(package_name.__version__)"
```

## Performance Tips

1. **Use scratch directories**: Set `APPTAINER_CACHEDIR` to `$SLURMTMPDIR`
2. **Request appropriate resources**: More CPUs speed up container builds
3. **Bind mount data**: Avoid copying large datasets into the container
4. **Use debug partition**: For testing, use the debug partition with shorter time limits

## Support

For issues with the container:
1. Check the container test output
2. Verify your HPC system supports Singularity/Apptainer
3. Ensure you have appropriate permissions and resources allocated
4. Check the original project README for specific usage instructions

## References

- [Original Project](https://github.com/jiayu-peng-lab/PerovskiteOrderingGCNNs)
- [Singularity Documentation](https://docs.sylabs.io/)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [SLURM Documentation](https://slurm.schedmd.com/) 