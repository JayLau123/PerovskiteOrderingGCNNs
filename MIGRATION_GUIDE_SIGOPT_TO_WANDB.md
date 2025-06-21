# Migration Guide: SigOpt to Weights & Biases (wandb)

This guide provides step-by-step instructions for migrating from SigOpt to Weights & Biases (wandb) for hyperparameter optimization and experiment tracking in the PerovskiteOrderingGCNNs project.

## Overview

Weights & Biases (wandb) offers several advantages over SigOpt:
- **Better visualization and monitoring** of training progress
- **Real-time experiment tracking** with rich dashboards
- **Improved collaboration** features for team projects
- **Artifact management** for model versioning
- **Free tier** with generous limits for academic use

## Step 1: Install wandb

### Update Environment
The `environment.yml` file has been updated to include wandb. Install it using:

```bash
conda env update -f environment.yml
```

Or install manually:
```bash
pip install wandb==0.16.3
```

### Login to wandb
```bash
wandb login
```

You'll need to create an account at [wandb.ai](https://wandb.ai) if you don't have one.

## Step 2: Key File Changes

### New Files Created:
1. **`training/hyperparameters/wandb_parameters.py`** - Hyperparameter sweep configurations
2. **`training/wandb_utils.py`** - Utility functions for wandb integration
3. **`training/run_wandb_experiment.py`** - Main training script with wandb integration
4. **`1_model_training_wandb.ipynb`** - Updated Jupyter notebook

### Files to Replace:
- **`training/run_sigopt_experiment.py`** → **`training/run_wandb_experiment.py`**
- **`1_model_training.ipynb`** → **`1_model_training_wandb.ipynb`**

## Step 3: Usage Examples

### Basic Hyperparameter Optimization

```python
from training.run_wandb_experiment import run_wandb_experiment

# Run hyperparameter optimization for CGCNN
run_wandb_experiment(
    struct_type="unrelaxed",
    model_type="CGCNN",
    gpu_num=0,
    obs_budget=50,  # Number of trials
    training_fraction=1.0,
    training_seed=0,
    project_name="perovskite_cgcnn"  # Optional custom project name
)
```

### Single Training Run

```python
from training.run_wandb_experiment import run_single_training_run

# Run a single training run with fixed hyperparameters
val_mae = run_single_training_run(
    struct_type="unrelaxed",
    model_type="CGCNN",
    gpu_num=0,
    training_fraction=1.0,
    training_seed=0,
    project_name="perovskite_test",
    run_name="cgcnn_baseline"
)
```

### Command Line Usage

```bash
# Hyperparameter optimization
python training/run_wandb_experiment.py \
    --model CGCNN \
    --struct_type unrelaxed \
    --gpu 0 \
    --budget 50 \
    --training_fraction 1.0 \
    --project_name my_perovskite_project

# Single training run
python training/run_wandb_experiment.py \
    --model CGCNN \
    --struct_type unrelaxed \
    --gpu 0 \
    --single_run \
    --run_name test_run \
    --project_name my_perovskite_project
```

## Step 4: Key Differences

### Project Organization
- **SigOpt**: Experiments were standalone
- **wandb**: Experiments are organized into projects for better management

### Hyperparameter Configuration
- **SigOpt**: Used `get_cgcnn_hyperparameter_range()` functions
- **wandb**: Uses sweep configurations in `wandb_parameters.py`

### Model Saving
- **SigOpt**: Models saved locally with manual file management
- **wandb**: Models automatically saved as artifacts with versioning

### Monitoring
- **SigOpt**: Limited real-time monitoring
- **wandb**: Rich real-time dashboards with training curves, hyperparameter comparisons, etc.

## Step 5: Retrieving Best Models

### Using wandb API
```python
from training.wandb_utils import download_best_model_artifacts

# Download the 3 best models from a project
best_models = download_best_model_artifacts(
    project_name="perovskite_cgcnn",
    model_type="CGCNN",
    num_best_models=3,
    save_dir="./best_models/"
)
```

### Manual Download
1. Go to your wandb project dashboard
2. Find the best run based on validation MAE
3. Download the model artifact from the run page

## Step 6: Updating Inference Pipeline

The inference pipeline (`2_model_inference.ipynb`) needs to be updated to work with wandb models. The main changes are:

1. **Model loading**: Models are now stored as wandb artifacts
2. **Path structure**: Model paths follow wandb conventions
3. **Metadata**: Hyperparameters are stored in wandb run configs

### Updated Inference Functions
```python
from training.wandb_utils import get_best_models_from_wandb

# Get best runs from wandb
best_runs = get_best_models_from_wandb(
    project_name="perovskite_cgcnn",
    model_type="CGCNN",
    num_best_models=3
)

# Use these runs for inference
for run in best_runs:
    # Download model and run inference
    # ... (implementation details)
```

## Step 7: Configuration Options

### Sweep Methods
Available sweep methods in wandb:
- `bayes`: Bayesian optimization (recommended)
- `grid`: Grid search
- `random`: Random search

### Early Termination
wandb supports early termination of poorly performing runs:
```python
sweep_config['early_terminate'] = {
    'type': 'hyperband',
    'min_iter': 10
}
```

### Parallel Execution
wandb supports parallel execution of multiple runs:
```python
# Run multiple agents in parallel
wandb.agent(sweep_id, function=train_function, count=50)
```

## Step 8: Migration Checklist

- [ ] Install wandb and update environment
- [ ] Login to wandb account
- [ ] Test single training run
- [ ] Run hyperparameter optimization
- [ ] Verify model artifacts are saved correctly
- [ ] Update inference pipeline
- [ ] Update documentation and notebooks
- [ ] Remove SigOpt dependencies (optional)

## Step 9: Troubleshooting

### Common Issues

1. **wandb login issues**
   ```bash
   # Check if logged in
   wandb status
   
   # Re-login if needed
   wandb login --relogin
   ```

2. **GPU memory issues**
   - Reduce batch size in hyperparameter ranges
   - Use smaller training fractions for testing

3. **Artifact download issues**
   - Check internet connection
   - Verify wandb API key is set correctly

4. **Sweep not starting**
   - Check sweep configuration syntax
   - Verify project name is valid

### Getting Help

- [wandb Documentation](https://docs.wandb.ai/)
- [wandb Community](https://community.wandb.ai/)
- [wandb GitHub](https://github.com/wandb/wandb)

## Step 10: Performance Comparison

### Advantages of wandb:
- **Better visualization**: Rich dashboards with training curves
- **Real-time monitoring**: Live updates during training
- **Artifact management**: Automatic model versioning
- **Collaboration**: Team access to experiments
- **Free tier**: Generous limits for academic use

### Migration Benefits:
- **Improved debugging**: Better visibility into training process
- **Easier comparison**: Side-by-side run comparisons
- **Version control**: Automatic tracking of model versions
- **Reproducibility**: Better experiment tracking

## Conclusion

The migration from SigOpt to wandb provides significant improvements in experiment tracking, visualization, and collaboration. The new system maintains the same functionality while offering enhanced monitoring and management capabilities.

For questions or issues during migration, refer to the wandb documentation or create an issue in the project repository. 