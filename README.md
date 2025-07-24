# PerovskiteOrderingGCNNs

[![arXiv](https://img.shields.io/badge/arXiv-2409.13851-red.svg)](https://arxiv.org/abs/2409.13851)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.13820311-blue.svg)](https://doi.org/10.5281/zenodo.13820311)
[![MDF](https://img.shields.io/badge/Materials_Data_Facility-10.18126/ncqt--rh18-purple.svg)](https://doi.org/10.18126/ncqt-rh18)
[![MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/license/mit)

Repo for our paper **"Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks"** ([preprint on arXiv](https://arxiv.org/abs/2409.13851)).

## 📥 Download

To start, clone this repo and all its submodules to your local directory or a workstation:
```
git clone --recurse-submodules git@github.com:jiayu-peng-lab/PerovskiteOrderingGCNNs.git
```
or
```
git clone git@github.com:jiayu-peng-lab/PerovskiteOrderingGCNNs.git
cd PerovskiteOrderingGCNNs
git submodule update --init
```

Our codes are built upon previous implementations of [CGCNN](https://github.com/-mit/PerovskiteOrderingGCNNs_cgcnn/tree/af4c0bf6606da1b46887ed8c29521d199d5e2798), [e3nn](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_e3nn/tree/408b90e922a2a9c7bae2ad95433aae97d1a58494), and [PaiNN](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_painn/tree/e7980a52af4936addc5fb03dbc50d4fc74fe98fc), which are included as submodules in this repo. If there are any changes in their corresponding GitHub repos, the following command will update the submodules in this repo:
```
git submodule update --remote --merge
```

---

## 🚀 Usage

### 🖥️ If working on a Linux workstation

To automatically download all required data and models and set up the conda environment, run:
```
bash scripts/setup_project.sh
```
This script will:
- Download and extract the datasets and model files from Zenodo.
- Set up the Conda environment (using the provided [environment.yml](environment.yml) if available).
- Ensure you are ready to run the notebooks or scripts.

Alternatively, you can download all our data and trained models manually; they are archived on Zenodo ([DOI: 10.5281/zenodo.13820311](https://doi.org/10.5281/zenodo.13820311)) and Materials Data Facility ([DOI: 10.18126/ncqt-rh18](https://doi.org/10.18126/ncqt-rh18)). Please place all data and model files in the corresponding directories and then refer to the following Jupyter notebooks below to reproduce the results of our paper. Moreover, if you want to install the Conda environment manually, this repository requires the following packages to run correctly:
```
pandas            1.5.3
scipy             1.10.1
numpy             1.24.3
scikit-learn      1.2.2
matplotlib        3.7.1
seaborn           0.12.2
pymatgen          2023.5.10
ase               3.22.1
rdkit             2023.3.1
e3fp              1.2.5
pytorch           1.13.1
pytorch-cuda      11.7
pytorch-sparse    0.6.17
pytorch-scatter   2.1.1
pytorch-cluster   1.6.1
torchvision       0.14.1
torchaudio        0.13.1
pyg               2.3.0
e3nn              0.5.1
wandb             0.16.3
gdown             4.7.1
mscorefonts       0.0.1
boken             3.3.4
```

All these packages can be installed using the [`environment.yml`](environment.yml) file and Conda:
```
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```

Afterwards, you can run the following three notebooks to reproduce the main results of this paper:
- [`1_model_training.ipynb`](1_model_training.ipynb): Train GCNNs and conduct hyperparameter optimization.
- [`2_model_inference.ipynb`](2_model_inference.ipynb): Verify performance, select top models, compute predictions, and extract latent embeddings.
- [`3_model_analysis.ipynb`](3_model_analysis.ipynb): Reproduce all major figures in the manuscript.

---

### 🖥️ If working on an HPC cluster

An example is provided here for running deep learning codes on HPC clusters (such as those in [UB CCR](https://www.buffalo.edu/ccr.html)) using containers (for Conda) and the Slurm job scheduler.

#### 1. Create a Singularity/Apptainer `.def` file from `environment.yml` or `requirements.txt`:

To set the Apptainer cache directory, run:

```bash
export APPTAINER_CACHEDIR="$(pwd)/.apptainer_cache"
```

Example `.def` file:

```text
Bootstrap: docker
From: continuumio/miniconda3:latest

%post
    # Create environment.yml directly in the container
    cat > /tmp/environment.yml << 'EOF'
name: perovskite_env
channels:
  - pyg
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10.11
  - pytorch=1.13.1
  - pytorch-cuda=11.7
  # ... additional dependencies
EOF
    
    # Create the conda environment
    conda env create -f /tmp/environment.yml
    conda clean -a
    
    # Make sure the environment is activated by default
    echo "source activate perovskite_env" >> ~/.bashrc

%environment
    export PATH=/opt/conda/envs/perovskite_env/bin:$PATH
    export CONDA_DEFAULT_ENV=perovskite_env

%runscript
    exec "$@"
```

#### 2. Run job on the UB CCR (two options):

##### 2A. Submitting Jobs

You can edit `scripts/run_container_job.sh` to set the desired model and dataset by changing the `MODEL` and `DATASET` variables in the script.

- `MODEL`: Specifies which graph neural network architecture to use. Set to either `CGCNN` or `e3nn`.
  - Example: `MODEL="CGCNN"` or `MODEL="e3nn"`
- `DATASET`: Specifies which dataset to use for the experiment. Set to either `relaxed` or `unrelaxed`.
  - Example: `DATASET="relaxed"` or `DATASET="unrelaxed"`

To change these, open `scripts/run_container_job.sh` in a text editor and modify the lines:

```bash
MODEL="CGCNN"         # or "e3nn"
DATASET="relaxed"     # or "unrelaxed"
```

The script also passes several command-line arguments to the Python experiment script:

- `--budget 50`: Number of hyperparameter optimization trials or max training runs.
- `--training_fraction 1`: Fraction of the available training data to use (1 = 100%).
- `--training_seed 0`: Random seed for reproducibility.
- `--gpu 0`: Which GPU to use inside the container.
- `--resume_sweep_id <sweep_id>`: (Optional) Resume a previous hyperparameter sweep.

To change these arguments, edit the last line of the script:

```bash
python training/run_wandb_experiment.py --struct_type $DATASET --model $MODEL --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
```

Change the values as needed for your experiment.

Once you have set the desired options, run the script with:

```bash
bash scripts/run_container_job.sh
```

##### 2B. Interactive Shell

Use this command to get the resource to perform the computation:

```bash
salloc --partition=general-compute --qos=general-compute --mem=64G --time=72:00:00 --gpus-per-node=1
```

Then, in your interactive session, follow these steps:

1. **(If needed) Load Apptainer/Singularity (module name may vary by cluster):**
   ```bash
   module load apptainer  # or singularity
   ```

2. **Navigate to your project directory (if not already there):**
   ```bash
   cd /path/to/PerovskiteOrderingGCNNs
   ```

3. **Start an interactive shell inside the container:**
   ```bash
   apptainer shell --nv --bind $(pwd):/workspace --pwd /workspace perovskite_exp.sif
   ```

4. **Activate the conda environment inside the container:**
   ```bash
   source activate perovskite_env
   ```

5. **(Optional) Set up cache directories for matplotlib, pip, etc.:**
   ```bash
   mkdir -p /workspace/cache/{matplotlib,pip,fontconfig}
   export MPLCONFIGDIR=/workspace/cache/matplotlib
   export PIP_CACHE_DIR=/workspace/cache/pip
   export FONTCONFIG_PATH=/workspace/cache/fontconfig
   export XDG_CACHE_HOME=/workspace/cache
   export PYTHONPATH=/workspace:/workspace/local_packages:$PYTHONPATH
   ```

6. **Run your experiment (example):**
   ```bash
   python training/run_wandb_experiment.py --struct_type relaxed --model CGCNN --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
   ```
   - Change `--struct_type` to `unrelaxed` and `--model` to `e3nn` as needed.
   - Adjust other arguments as desired (see above for details).

Once you have set the desired options, run your experiment with:

```bash
python training/run_wandb_experiment.py --struct_type relaxed --model CGCNN --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
```

7. **To resume a previous sweep, add the `--resume_sweep_id <sweep_id>` argument.**

This approach allows you to run and debug your experiments interactively, monitor outputs in real time, and make adjustments as needed.

---

## 📖 Citation

If you use our codes, data, and/or models, please cite the following paper:

```bibtex
@article{peng2024learning,
  title={Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks},
  author={Jiayu Peng and James Damewood and Jessica Karaguesian and Jaclyn R. Lunger and Rafael Gómez-Bombarelli},
  journal={arXiv:2409.13851},
  url = {https://arxiv.org/abs/2409.13851},
  year={2024}
}
```
