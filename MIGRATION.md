# SigOpt to Weights & Biases (wandb) Migration Summary

This document summarizes the conversion from SigOpt to Weights & Biases (wandb) for the PerovskiteOrderingGCNNs project.

## Overview

The migration involved creating new wandb-equivalent files while preserving the original SigOpt files for reference. All new files follow the same structure and functionality as their SigOpt counterparts, with the key difference being the use of wandb for experiment tracking and hyperparameter optimization.

## Files Created

### 1. `training/wandb_utils.py`
**Purpose**: Wandb utilities equivalent to `sigopt_utils.py`

**Key Functions**:
- `build_wandb_name()` - Creates consistent naming for wandb experiments using the same logic as SigOpt

**Changes**:
- Replaced `build_sigopt_name()` with `build_wandb_name()`
- Maintained identical naming logic and parameter handling
- Added documentation explaining the function's purpose

### 2. `training/hyperparameters/wandb_parameters.py`
**Purpose**: Wandb hyperparameter configurations equivalent to `sigopt_parameters.py`

**Key Functions**:
- `get_cgcnn_hyperparameter_range()` - Wandb sweep configuration for CGCNN
- `get_painn_hyperparameter_range()` - Wandb sweep configuration for Painn
- `get_e3nn_hyperparameter_range()` - Wandb sweep configuration for e3nn
- `convert_hyperparameters()` - Converts wandb format to expected training format

**Changes**:
- Converted SigOpt parameter definitions to wandb sweep configurations
- Used `bayes` method for Bayesian optimization (equivalent to SigOpt's default)
- Added proper wandb parameter types (`int_uniform`, `values`, etc.)
- Original SigOpt functions commented out for reference

**Configuration Differences**:
```python
# SigOpt format (commented out)
# dict(name="batch_size", bounds=dict(min=4, max=16), type="int")

# Wandb format (active)
'batch_size': {
    'values': [4, 8, 12, 16]
}
```

### 3. `training/run_wandb_experiment.py`
**Purpose**: Main wandb experiment runner equivalent to `run_sigopt_experiment.py`

**Key Functions**:
- `run_wandb_experiment()` - Main experiment runner using wandb sweeps
- `wandb_evaluate_model()` - Model evaluation function
- `create_wandb_experiment()` - Creates wandb sweep configuration

**Changes**:
- Replaced SigOpt experiment creation with wandb sweep setup
- Used `wandb.sweep()` and `wandb.agent()` for hyperparameter optimization
- Maintained same data processing and model training logic
- Original SigOpt code commented out with explanatory comments

**Key Differences**:
```python
# SigOpt approach (commented out)
# conn = sigopt.Connection(driver="lite")
# experiment = conn.experiments().create(...)
# suggestion = conn.experiments(experiment.id).suggestions().create()

# Wandb approach (active)
sweep_config = create_wandb_experiment(...)
sweep_id = wandb.sweep(sweep_config, project="perovskite-ordering-gcnns")
wandb.agent(sweep_id, train_function, count=obs_budget)
```

### 4. `inference/select_best_models_wandb.py`
**Purpose**: Wandb version of model selection equivalent to `select_best_models.py`

**Key Functions**:
- `get_experiment_id()` - Gets wandb experiment ID
- `load_model()` - Loads trained models
- `reverify_wandb_models()` - Reverifies model performance
- `keep_the_best_few_models()` - Selects best performing models

**Changes**:
- Replaced `build_sigopt_name()` with `build_wandb_name()`
- Updated file paths to use wandb naming convention
- Maintained same model loading and evaluation logic
- Original SigOpt connection code commented out

### 5. `inference/embedding_extraction_wandb.py`
**Purpose**: Wandb version of embedding extraction equivalent to `embedding_extraction.py`

**Key Functions**:
- `extract_embeddings()` - Extracts model embeddings
- `get_experiment_id()` - Gets wandb experiment ID

**Changes**:
- Updated to use wandb naming convention
- Maintained same embedding extraction logic
- Updated file paths for model loading

### 6. `inference/test_model_prediction_wandb.py`
**Purpose**: Wandb version of model prediction testing equivalent to `test_model_prediction.py`

**Key Functions**:
- `test_model_prediction()` - Tests model predictions
- `get_experiment_id()` - Gets wandb experiment ID

**Changes**:
- Updated to use wandb naming convention
- Maintained same prediction testing logic
- Updated file paths for model loading and result saving

### 7. `inference/plot_utils_wandb.py`
**Purpose**: Wandb version of plotting utilities equivalent to `plot_utils.py`

**Key Functions**:
- `plot_training_curves()` - Plots training curves
- `plot_predictions_vs_targets()` - Plots predictions vs targets
- `get_experiment_id()` - Gets wandb experiment ID

**Changes**:
- Updated to use wandb naming convention
- Maintained same plotting logic and functionality
- Updated file paths for data loading

## Key Migration Patterns

### 1. Import Changes
```python
# Original SigOpt imports (commented out)
# import sigopt
# from training.sigopt_utils import build_sigopt_name

# New wandb imports (active)
import wandb
from training.wandb_utils import build_wandb_name
```

### 2. Naming Convention
```python
# Original SigOpt naming (commented out)
# sigopt_name = build_sigopt_name(...)

# New wandb naming (active)
wandb_name = build_wandb_name(...)
```

### 3. Experiment Management
```python
# SigOpt experiment creation (commented out)
# conn = sigopt.Connection(driver="lite")
# experiment = conn.experiments().create(...)

# Wandb sweep creation (active)
sweep_config = create_wandb_experiment(...)
sweep_id = wandb.sweep(sweep_config, project="perovskite-ordering-gcnns")
```

### 4. Hyperparameter Optimization
```python
# SigOpt suggestion loop (commented out)
# suggestion = conn.experiments(experiment.id).suggestions().create()
# hyperparameters = suggestion.assignments

# Wandb agent approach (active)
def train_function():
    wandb.init()
    hyperparameters = dict(wandb.config)
    # ... training logic ...
    wandb.log({"val_mae": val_loss})

wandb.agent(sweep_id, train_function, count=obs_budget)
```

## Benefits of Migration

1. **Enhanced Experiment Tracking**: Wandb provides real-time monitoring and visualization
2. **Better Collaboration**: Team members can easily view and compare experiments
3. **Improved Visualization**: Built-in plotting and dashboard capabilities
4. **Version Control**: Better integration with model versioning
5. **Cost Efficiency**: Wandb offers more generous free tier compared to SigOpt

## Usage Instructions

1. **Setup**: Ensure wandb is installed and configured (`pip install wandb`)
2. **Authentication**: Run `wandb login` to authenticate
3. **Training**: Use `run_wandb_experiment()` instead of `run_sigopt_experiment()`
4. **Inference**: Use the `_wandb` versions of inference files
5. **Monitoring**: View experiments in the wandb dashboard

## File Structure

```
training/
├── wandb_utils.py                    # New
├── hyperparameters/
│   └── wandb_parameters.py          # New
└── run_wandb_experiment.py          # New

inference/
├── select_best_models_wandb.py      # New
├── embedding_extraction_wandb.py    # New
├── test_model_prediction_wandb.py   # New
└── plot_utils_wandb.py             # New
```

## Notes

- All original SigOpt files remain unchanged for reference
- New wandb files maintain identical functionality
- Extensive comments explain the changes and differences
- The migration preserves the existing workflow while leveraging wandb's capabilities
- Both systems can coexist during the transition period

This migration provides a smooth transition from SigOpt to wandb while maintaining all existing functionality and improving the overall experiment management experience. 