# PerovskiteOrderingGCNNs

[![arXiv](https://img.shields.io/badge/arXiv-2409.13851-red.svg)](https://arxiv.org/abs/2409.13851)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.13820311-blue.svg)](https://doi.org/10.5281/zenodo.13820311)
[![MDF](https://img.shields.io/badge/Materials_Data_Facility-10.18126/ncqt--rh18-purple.svg)](https://doi.org/10.18126/ncqt-rh18)
[![MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/license/mit)

Repo for our paper "Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks" ([preprint on arXiv](https://arxiv.org/abs/2409.13851)).

---

## 🚀 Quick Start

### 1. Clone the repository (with submodules)
```bash
git clone --recurse-submodules git@github.com:learningmatter-mit/PerovskiteOrderingGCNNs.git
cd PerovskiteOrderingGCNNs
git submodule update --init
```

### 2. Set up the environment
Install all dependencies using conda:
```bash
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```

### 3. Prepare data
Download data and pre-trained models from [Zenodo](https://doi.org/10.5281/zenodo.13820311) or [Materials Data Facility](https://doi.org/10.18126/ncqt-rh18) and place them in the appropriate directories (`data/`, `best_models/`, etc.).

---

## 🗂️ Directory Structure (Updated)

```
/ (root)
│   README.md
│   LICENSE
│   .gitignore
│   .gitmodules
│
├── notebooks/
│     1_model_training.ipynb
│     2_model_inference.ipynb
│     3_model_analysis.ipynb
│
├── src/
│     1_model_training.py
│     2_model_inference.py
│     3_model_analysis.py
│     train_with_wandb.py
│
├── scripts/
│     train.py
│     infer.py
│     analyze.py
│     ...
│
├── docker/
│     Dockerfile
│     docker-compose.yml
│
├── config/
│     environment.yml
│
├── docs/
│     MIGRATION_GUIDE_SIGOPT_TO_WANDB.md
│
├── models/
├── data/
├── figures/
├── inference/
├── processing/
├── training/
├── wandb/
├── best_models/
├── saved_models/
```

- All code scripts are now in `/src/`.
- All Jupyter notebooks are in `/notebooks/`.
- Docker-related files are in `/docker/`.
- Environment and config files are in `/config/`.
- Documentation and guides are in `/docs/`.
- The `scripts/`, `models/`, `data/`, `figures/`, `inference/`, `processing/`, `training/`, `wandb/`, `best_models/`, and `saved_models/` directories remain as they are.

---

## 🏃 Usage (Updated)

### Training
Train a model with hyperparameter optimization:
```bash
python scripts/train.py --model_type CGCNN --struct_type unrelaxed --gpu_num 0 --obs_budget 10 --training_fraction 1.0 --training_seed 0
```

### Inference & Embedding Extraction
```bash
python scripts/infer.py --model_type CGCNN --struct_type unrelaxed --gpu_num 0 --training_fraction 1.0 --num_best_models 3 --reverify --select_best --predict --embeddings
```

### Analysis & Plotting
```bash
python scripts/analyze.py --all
```

---

## 📚 Submodules & Advanced Usage

- **CGCNN**: See `models/PerovskiteOrderingGCNNs_cgcnn/README.md` for details on custom datasets and standalone usage.
- **e3nn**: See `models/PerovskiteOrderingGCNNs_e3nn/README.md` for symmetry-aware neural network details.
- **PaiNN/NFF**: See `models/PerovskiteOrderingGCNNs_painn/README.md` and `scripts/cp3d/README.md` for advanced property prediction, force fields, and 3D molecular tasks.

---

## 🧑‍💻 Development & Contribution
- All main workflows are now accessible via CLI scripts in `scripts/`.
- Notebooks are provided for reference and legacy reproducibility, but all new work should use the scripts.
- For submodule-specific development, see their respective READMEs.

---

## 📖 Citation
If you use our codes, data, and/or models, please cite:
```bibtex
@article{peng2024learning,
  title={Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks},
  author={Jiayu Peng and James Damewood and Jessica Karaguesian and Jaclyn R. Lunger and Rafael Gómez-Bombarelli},
  journal={arXiv:2409.13851},
  url = {https://arxiv.org/abs/2409.13851},
  year={2024}
}
```

---

## 🔗 References
- [CGCNN Paper](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)
- [e3nn Paper](https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214)
- [PaiNN Paper](https://arxiv.org/abs/2102.03150)
- [NFF Documentation](https://github.com/learningmatter-mit/NeuralForceField)

---

## 🐳 Docker & Docker Compose Features (Updated)

- The Dockerfile and docker-compose.yml are now in `/docker/`.
- Use the same commands as before, but reference the new locations if building manually:
  ```bash
  docker build -t perovskite-gcnns -f docker/Dockerfile .
  docker-compose -f docker/docker-compose.yml build
  ```

| Feature         | Dockerfile | docker-compose | Description |
|----------------|------------|---------------|-------------|
| CLI scripts    | ✅         | ✅            | Run all CLI workflows in a container |
| GPU support    | ✅         | ✅            | Use `--gpus all` (Docker) or enable in compose for GPU |
| Jupyter        | ✅         | ✅            | Expose port 8888 for notebooks |
| Volume mount   | ✅         | ✅            | Mounts your code/data for persistence |
| Multi-service  |            | ✅            | Separate app & Jupyter services |

### Using docker-compose

#### 1. Build all services
```bash
docker-compose build
```

#### 2. Run the main CLI environment (CPU)
```bash
docker-compose run --rm app
```

#### 3. Run with GPU (NVIDIA Docker)
```bash
docker-compose run --rm --gpus all app
```

#### 4. Start a Jupyter notebook server
```bash
docker-compose up jupyter
```
- Access at [http://localhost:8888](http://localhost:8888)

#### 5. Run CLI scripts inside the container
Once inside the `app` service shell:
```bash
python scripts/train.py --help
python scripts/infer.py --help
python scripts/analyze.py --help
```

#### 6. Enable GPU in compose (alternative)
Uncomment the `deploy` or `runtime: nvidia` lines in `docker-compose.yml` as needed for your Docker version.

---
