import os
import json
import wandb
import shutil
from datetime import datetime


def build_wandb_name(data_name, target_prop, struct_type, interpolation, model_type, contrastive_weight, training_fraction, training_seed):
    """Build a consistent name for wandb experiments"""
    interpolation_str = "_interpolation" if interpolation else ""
    contrastive_str = f"_contrastive{contrastive_weight}" if contrastive_weight != 1.0 else ""
    fraction_str = f"_TrainingFraction{training_fraction}" if training_fraction != 1.0 else ""
    seed_str = f"_Seed{training_seed}" if training_seed != 0 else ""
    
    name = f"{target_prop}_htvs_data_{struct_type}_{model_type}{interpolation_str}{contrastive_str}{fraction_str}{seed_str}"
    return name


def setup_wandb_run(data_name, target_prop, struct_type, interpolation, model_type, contrastive_weight, training_fraction, training_seed, hyperparameters, project_name="perovskite-ordering-gcnns"):
    """Setup a new wandb run with proper configuration"""
    run_name = build_wandb_name(data_name, target_prop, struct_type, interpolation, model_type, contrastive_weight, training_fraction, training_seed)
    
    # Convert hyperparameters for logging
    config = {
        'data_name': data_name,
        'target_prop': target_prop,
        'struct_type': struct_type,
        'interpolation': interpolation,
        'model_type': model_type,
        'contrastive_weight': contrastive_weight,
        'training_fraction': training_fraction,
        'training_seed': training_seed,
        **hyperparameters
    }
    
    # Initialize wandb run
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=[model_type, struct_type, target_prop]
    )
    
    return run


def save_model_to_wandb(model_path, model_type, run_name, run_id, observation_count=None):
    """Save model as wandb artifact"""
    if observation_count is not None:
        artifact_name = f"{run_name}_obs_{observation_count}"
    else:
        artifact_name = f"{run_name}_final"
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"{model_type} model trained on {run_name}"
    )
    
    # Add model files to artifact
    if os.path.exists(model_path):
        if os.path.isfile(model_path):
            artifact.add_file(model_path)
        else:
            # If it's a directory, add all files
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    artifact.add_file(file_path, name=os.path.relpath(file_path, model_path))
    
    # Log the artifact
    wandb.log_artifact(artifact)
    
    return artifact


def log_training_metrics(epoch, train_loss, val_loss, learning_rate, additional_metrics=None):
    """Log training metrics to wandb"""
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    wandb.log(metrics)


def log_hyperparameters(hyperparameters):
    """Log hyperparameters to wandb"""
    wandb.config.update(hyperparameters)


def get_best_models_from_wandb(project_name="perovskite-ordering-gcnns", model_type=None, struct_type=None, target_prop=None, limit=10):
    """Get best models from wandb based on validation loss"""
    api = wandb.Api()
    
    # Build query filters
    filters = []
    if model_type:
        filters.append({"tags": {"$in": [model_type]}})
    if struct_type:
        filters.append({"config.struct_type": struct_type})
    if target_prop:
        filters.append({"config.target_prop": target_prop})
    
    # Get runs
    runs = api.runs(project_name, filters={"$and": filters} if filters else None)
    
    # Sort by validation loss
    sorted_runs = sorted(runs, key=lambda run: run.summary.get('val_mae', float('inf')))
    
    return sorted_runs[:limit]


def download_best_model_artifacts(project_name="perovskite-ordering-gcnns", model_type=None, struct_type=None, target_prop=None, download_dir="./downloaded_models"):
    """Download best model artifacts from wandb"""
    best_runs = get_best_models_from_wandb(project_name, model_type, struct_type, target_prop, limit=5)
    
    downloaded_models = []
    
    for i, run in enumerate(best_runs):
        # Get artifacts for this run
        artifacts = run.logged_artifacts()
        
        for artifact in artifacts:
            if artifact.type == "model":
                # Download artifact
                artifact_dir = os.path.join(download_dir, f"{run.name}_{i}")
                os.makedirs(artifact_dir, exist_ok=True)
                
                artifact.download(root=artifact_dir)
                downloaded_models.append({
                    'run_name': run.name,
                    'run_id': run.id,
                    'val_mae': run.summary.get('val_mae', 'N/A'),
                    'artifact_name': artifact.name,
                    'local_path': artifact_dir
                })
    
    return downloaded_models


def create_sweep_config(model_type, project_name="perovskite-ordering-gcnns", sweep_name=None):
    """Create a sweep configuration for hyperparameter optimization"""
    from training.hyperparameters.wandb_parameters import get_sweep_config
    
    sweep_config = get_sweep_config(model_type)
    
    # Add project and program info
    sweep_config.update({
        'project': project_name,
        'program': 'training/run_wandb_experiment.py'
    })
    
    if sweep_name:
        sweep_config['name'] = sweep_name
    
    return sweep_config


def cleanup_temp_directories(temp_dir):
    """Clean up temporary directories after training"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def log_model_summary(model, model_type):
    """Log model summary to wandb"""
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        })
        
        print(f"Model Summary - Total params: {total_params:,}, Trainable params: {trainable_params:,}") 