# PerovskiteOrderingGCNNs

[![arXiv](https://img.shields.io/badge/arXiv-2409.13851-red.svg)](https://arxiv.org/abs/2409.13851)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.13820311-blue.svg)](https://doi.org/10.5281/zenodo.13820311)
[![MDF](https://img.shields.io/badge/Materials_Data_Facility-10.18126/ncqt--rh18-purple.svg)](https://doi.org/10.18126/ncqt-rh18)
[![MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/license/mit)

Repo for our paper "Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks" ([preprint on arXiv](https://arxiv.org/abs/2409.13851)).

## Setup

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

This repository requires the following packages to run correctly:

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

All these packages can be installed using the [environment.yml](environment.yml) file and `conda`:

```
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```

## Usage

All our data and trained models are archived on Zenodo ([DOI: 10.5281/zenodo.13820311](https://doi.org/10.5281/zenodo.13820311)) and Materials Data Facility ([DOI: 10.18126/ncqt-rh18](https://doi.org/10.18126/ncqt-rh18)). Please place all data and model files in the corresponding directories and then refer to the following Jupyter notebooks to reproduce the results of our paper:

- [1_model_training.ipynb](1_model_training.ipynb): This notebook provides examples of how to train GCNNs on the training dataset and conduct hyperparameter optimization based on the loss on the validation set.
- [2_model_inference.ipynb](2_model_inference.ipynb): This notebook provides examples of how to verify the performance of GCNNs on the validation set, select the top-performing models accordingly, compute the prediction on the test and holdout sets, and extract the latent embeddings of CGCNN and e3nn after all message passing and graph convolution layers.
- [3_model_analysis.ipynb](3_model_analysis.ipynb): This notebook provides examples of how to reproduce all major figures in this manuscript.

## Scripted Setup and HPC Usage

### 1. Automated Project Setup (Recommended)

To automatically download all required data and models, and set up the conda environment, run:

```bash
bash scripts/setup_project.sh
```

This script will:
- Download and extract the datasets and model files from Zenodo.
- Set up the conda environment (using the provided environment.yml if available).
- Ensure you are ready to run the notebooks or scripts.

### 2. Running Jupyter Notebooks Locally

After setup, activate the environment and launch Jupyter:

```bash
conda activate Perovskite_ML_Environment
jupyter notebook
```

Then open the desired notebook (e.g., `1_model_training.ipynb`) in your browser.

### 3. Building and Using Containers for HPC (UB CCR or similar)

#### a. Build the Singularity/Apptainer Container

On a system with Singularity/Apptainer installed, run:

```bash
bash scripts/build_container.sh
```

This will create a container file (e.g., `perovskite_exp.sif`).

#### b. Submitting a Job on UB CCR (or other SLURM-based HPC)

Edit `scripts/run_container_job.sh` to set your email and any module loads required by your cluster. Then submit the job:

```bash
sbatch scripts/run_container_job.sh
```

This script will:
- Pull the required container if not present.
- Bind your data and output directories.
- Run the training notebook or script inside the container with GPU support.

**To run a different notebook or script, modify the relevant line in `run_container_job.sh` (see comments in the script).**

#### c. Example: Running a Python Script in the Container (Interactive)

```bash
apptainer exec --nv perovskite_exp.sif python your_script.py
```

Or for Jupyter:

```bash
apptainer exec --nv perovskite_exp.sif jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### d. Submitting Only One Job from the Job Array

By default, `scripts/run_container_job.sh` uses a SLURM job array to run all combinations of model and dataset. If you want to submit only one specific job (e.g., just CGCNN on relaxed), follow these steps:

1. **Edit the Array Line**
   
   Find this line in `scripts/run_container_job.sh`:
   ```bash
   #SBATCH --array=0-3
   ```
   Change it to the specific job index you want to run. For example, to run only the first job:
   ```bash
   #SBATCH --array=0
   ```

2. **Job Index Mapping**
   The mapping of job indices to model/dataset combinations is:
   - `0`: CGCNN, relaxed
   - `1`: CGCNN, unrelaxed
   - `2`: e3nn, relaxed
   - `3`: e3nn, unrelaxed

   For example, to run only e3nn on relaxed, use:
   ```bash
   #SBATCH --array=2
   ```

3. **Submit the Job**
   After editing, submit as usual:
   ```bash
   sbatch scripts/run_container_job.sh
   ```

**Tip:** You can also run a range or a comma-separated list, e.g. `#SBATCH --array=1,3` to run only jobs 1 and 3.

---

For more details, see the comments in each script in the `scripts/` directory.

## Citation

If you use our codes, data, and/or models, please cite the following paper:

```
@article{peng2024learning,
  title={Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks},
  author={Jiayu Peng and James Damewood and Jessica Karaguesian and Jaclyn R. Lunger and Rafael Gómez-Bombarelli},
  journal={arXiv:2409.13851},
  url = {https://arxiv.org/abs/2409.13851},
  year={2024}
```