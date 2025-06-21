import os
import json
import sys
import pandas as pd
import argparse
import pickle as pkl
import torch
import numpy as np
import random
import shutil
import wandb
from processing.utils import filter_data_by_properties, select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from processing.create_model.create_model import create_model
from training.hyperparameters.wandb_parameters import convert_hyperparameters
from training.model_training.trainer import *
from training.wandb_utils import *
from training.evaluate import *


def run_wandb_experiment(struct_type, model_type, gpu_num, obs_budget=50, training_fraction=1.0, 
                        data_name="data/", target_prop="dft_e_hull", interpolation=False, 
                        contrastive_weight=1.0, training_seed=0, nickname="", project_name="perovskite-ordering-gcnns"):
    """
    Run hyperparameter optimization using wandb sweeps
    
    Parameters:
    - struct_type: the structure representation to use (options: unrelaxed, relaxed, M3Gnet_relaxed)
    - model_type: the model architecture to use (options: CGCNN, e3nn, Painn)
    - gpu_num: the GPU to use
    - obs_budget: the budget of hyperparameter optimization
    - training_fraction: if not trained on the entire training set, the fraction of the training set to use
    - training_seed: if not trained on the entire training set, the random seed for selecting this fraction
    """
    
    # Load and process data
    training_data, validation_data, edge_data = load_data(data_name, training_fraction, training_seed, interpolation)
    
    # Process data
    processed_data = process_data([training_data, validation_data], target_prop, struct_type, interpolation)
    
    print("Completed data processing")
    
    # Create sweep configuration
    sweep_config = create_sweep_config(model_type, project_name)
    sweep_config['method'] = 'bayes'
    sweep_config['metric']['name'] = 'val_mae'
    sweep_config['metric']['goal'] = 'minimize'
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created wandb sweep with ID: {sweep_id}")
    
    # Define the training function for the sweep
    def train_function():
        # Initialize wandb run
        wandb.init()
        
        # Get hyperparameters from wandb
        hyperparameters = dict(wandb.config)
        
        # Convert hyperparameters to expected format
        hyperparameters = convert_hyperparameters(hyperparameters)
        
        # Train model
        val_loss = train_single_run(
            hyperparameters, processed_data, target_prop, interpolation, 
            struct_type, model_type, contrastive_weight, training_fraction, 
            training_seed, gpu_num, nickname, None
        )
        
        # Log final validation loss
        wandb.log({"val_mae": val_loss})
    
    # Run the sweep
    wandb.agent(sweep_id, train_function, count=obs_budget)
    
    print(f"Completed wandb sweep with {obs_budget} observations")


def train_single_run(hyperparameters, processed_data, target_prop, interpolation, struct_type, 
                    model_type, contrastive_weight, training_fraction, training_seed, gpu_num, 
                    nickname, wandb_run=None):
    """
    Train a single model with given hyperparameters
    """
    device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
    
    train_data = processed_data[0]
    validation_data = processed_data[1]
    
    per_site = "per_site" in target_prop
    
    # Create data loaders
    train_loader = get_dataloader(train_data, target_prop, model_type, hyperparameters["batch_size"], 
                                 interpolation, per_site=per_site)
    train_eval_loader = None
    
    if "e3nn" in model_type and "pretrain" not in "data/" and "per_site" not in target_prop:
        train_eval_loader = get_dataloader(train_data, target_prop, "e3nn_contrastive", 1, 
                                         interpolation, per_site=per_site)
        val_loader = get_dataloader(validation_data, target_prop, "e3nn_contrastive", 1, 
                                  interpolation, per_site=per_site)
    else:
        val_loader = get_dataloader(validation_data, target_prop, model_type, 1, 
                                  interpolation, per_site=per_site)
    
    # Create model
    model, normalizer = create_model(model_type, train_loader, interpolation, target_prop, 
                                   hyperparameters=hyperparameters, per_site=per_site)
    
    # Log model summary to wandb
    if wandb_run:
        log_model_summary(model, model_type)
    
    # Create temporary directory for model saving
    run_name = build_wandb_name("data/", target_prop, struct_type, interpolation, model_type, 
                               contrastive_weight, training_fraction, training_seed)
    model_tmp_dir = f'./saved_models/{model_type}/{run_name}/{wandb_run.id if wandb_run else "local"}/{nickname}_tmp{gpu_num}'
    
    if os.path.exists(model_tmp_dir):
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir)
    
    # Train model (note: trainer function doesn't accept wandb_run parameter, so we'll modify it later if needed)
    best_model, loss_fn = trainer(model, normalizer, model_type, train_loader, val_loader, 
                                 hyperparameters, model_tmp_dir, gpu_num, 
                                 train_eval_loader=train_eval_loader, 
                                 contrastive_weight=contrastive_weight)
    
    # Evaluate model
    is_contrastive = "contrastive" in model_type
    _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, 
                                   gpu_num, is_contrastive=is_contrastive, 
                                   contrastive_weight=contrastive_weight)
    
    # Save model to wandb if run is available
    if wandb_run:
        save_model_to_wandb(model_tmp_dir, model_type, run_name, wandb_run.id)
    
    # Clean up temporary directory
    cleanup_temp_directories(model_tmp_dir)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if model_type == "Painn":
        return best_loss
    else:
        return best_loss[0]


def load_data(data_name, training_fraction, training_seed, interpolation):
    """Load training, validation, and edge data"""
    if data_name == "data/":
        training_data = pd.read_json(data_name + 'training_set.json')
        training_data = training_data.sample(frac=training_fraction, replace=False, random_state=training_seed)
        validation_data = pd.read_json(data_name + 'validation_set.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')
        
        if not interpolation:
            training_data = pd.concat((training_data, edge_data))
    else:
        raise ValueError("Specified Data Directory Does Not Exist!")
    
    print("Loaded data")
    return training_data, validation_data, edge_data


def process_data(data_list, target_prop, struct_type, interpolation):
    """Process training and validation data"""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    processed_data = []
    
    for dataset in data_list:
        dataset = filter_data_by_properties(dataset, target_prop)
        dataset = select_structures(dataset, struct_type)
        
        if interpolation:
            dataset = apply_interpolation(dataset, target_prop)
        
        processed_data.append(dataset)
    
    return processed_data


def get_default_hyperparameters(model_type):
    """Get default hyperparameters for a model type"""
    if model_type == "Painn":
        from training.hyperparameters.default import get_default_painn_hyperparameters
        return get_default_painn_hyperparameters()
    elif model_type == "CGCNN":
        from training.hyperparameters.default import get_default_cgcnn_hyperparameters
        return get_default_cgcnn_hyperparameters()
    elif model_type == "e3nn":
        from training.hyperparameters.default import get_default_e3nn_hyperparameters
        return get_default_e3nn_hyperparameters()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_single_training_run(struct_type, model_type, gpu_num, training_fraction=1.0, 
                           data_name="data/", target_prop="dft_e_hull", interpolation=False, 
                           contrastive_weight=1.0, training_seed=0, project_name="perovskite-ordering-gcnns"):
    """
    Run a single training run with default hyperparameters (no sweep)
    """
    # Load and process data
    training_data, validation_data, edge_data = load_data(data_name, training_fraction, training_seed, interpolation)
    processed_data = process_data([training_data, validation_data], target_prop, struct_type, interpolation)
    
    print("Completed data processing")
    
    # Get default hyperparameters
    hyperparameters = get_default_hyperparameters(model_type)
    
    # Setup wandb run
    run = setup_wandb_run(data_name, target_prop, struct_type, interpolation, model_type, 
                         contrastive_weight, training_fraction, training_seed, hyperparameters, project_name)
    
    try:
        # Train model
        val_loss = train_single_run(hyperparameters, processed_data, target_prop, interpolation, 
                                  struct_type, model_type, contrastive_weight, training_fraction, 
                                  training_seed, gpu_num, "", run)
        
        print(f"Training completed. Final validation loss: {val_loss}")
        return val_loss
        
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for perovskite ordering GCNNs using wandb')
    parser.add_argument('--data_name', default="data/", type=str, metavar='name',
                        help="the source of the data")
    parser.add_argument('--prop', default="dft_e_hull", type=str, metavar='name',
                        help="the property to predict (default: dft_e_hull; other options: Op_band_center)")
    parser.add_argument('--struct_type', default='unrelaxed', type=str, metavar='struct_type',
                        help="using which structure representation (default: unrelaxed; other options: relaxed, M3Gnet_relaxed)")
    parser.add_argument('--interpolation', default='no', type=str, metavar='yes/no',
                        help="using interpolation (default: no; other options: yes)")
    parser.add_argument('--model_type', default='CGCNN', type=str, metavar='model_type',
                        help="model architecture (default: CGCNN; other options: e3nn, Painn)")
    parser.add_argument('--gpu_num', default=0, type=int, metavar='gpu_num',
                        help="GPU number to use")
    parser.add_argument('--obs_budget', default=50, type=int, metavar='obs_budget',
                        help="number of hyperparameter optimization trials")
    parser.add_argument('--training_fraction', default=1.0, type=float, metavar='training_fraction',
                        help="fraction of training data to use")
    parser.add_argument('--training_seed', default=0, type=int, metavar='training_seed',
                        help="random seed for training data selection")
    parser.add_argument('--contrastive_weight', default=1.0, type=float, metavar='contrastive_weight',
                        help="weight for contrastive loss")
    parser.add_argument('--project_name', default="perovskite-ordering-gcnns", type=str, metavar='project_name',
                        help="wandb project name")
    parser.add_argument('--single_run', action='store_true',
                        help="run single training instead of hyperparameter sweep")
    
    args = parser.parse_args()
    
    interpolation = args.interpolation.lower() == 'yes'
    
    if args.single_run:
        run_single_training_run(
            struct_type=args.struct_type,
            model_type=args.model_type,
            gpu_num=args.gpu_num,
            training_fraction=args.training_fraction,
            data_name=args.data_name,
            target_prop=args.prop,
            interpolation=interpolation,
            contrastive_weight=args.contrastive_weight,
            training_seed=args.training_seed,
            project_name=args.project_name
        )
    else:
        run_wandb_experiment(
            struct_type=args.struct_type,
            model_type=args.model_type,
            gpu_num=args.gpu_num,
            obs_budget=args.obs_budget,
            training_fraction=args.training_fraction,
            data_name=args.data_name,
            target_prop=args.prop,
            interpolation=interpolation,
            contrastive_weight=args.contrastive_weight,
            training_seed=args.training_seed,
            project_name=args.project_name
        ) 