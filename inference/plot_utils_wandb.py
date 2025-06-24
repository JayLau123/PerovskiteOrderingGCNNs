import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)

def plot_training_curves(model_params, target_prop="dft_e_hull"):
    """
    Plot training curves using wandb models - equivalent to SigOpt version
    """
    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    exp_id = get_experiment_id(model_params, target_prop)
    directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" +str(exp_id) + "/" + "best_"

    # Load training logs
    training_curves = []
    for idx in range(3):
        log_file = directory + str(idx) + "/log_human_read.csv"
        if os.path.exists(log_file):
            log_data = pd.read_csv(log_file)
            training_curves.append(log_data)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    for idx, curve in enumerate(training_curves):
        plt.subplot(2, 2, 1)
        plt.plot(curve['epoch'], curve['train_loss'], label=f'Model {idx}')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(curve['epoch'], curve['val_loss'], label=f'Model {idx}')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(curve['epoch'], curve['train_mae'], label=f'Model {idx}')
        plt.title('Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(curve['epoch'], curve['val_mae'], label=f'Model {idx}')
        plt.title('Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    save_directory = "./plots/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    plt.savefig(save_directory + "/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_directory}")


def plot_predictions_vs_targets(model_params, target_prop="dft_e_hull"):
    """
    Plot predictions vs targets using wandb models - equivalent to SigOpt version
    """
    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    exp_id = get_experiment_id(model_params, target_prop)
    
    # Load predictions
    predictions_file = "./predictions/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id) + "/predictions.npy"
    targets_file = "./predictions/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id) + "/targets.npy"
    
    if os.path.exists(predictions_file) and os.path.exists(targets_file):
        predictions = np.load(predictions_file)
        targets = np.load(targets_file)
        
        plt.figure(figsize=(15, 5))
        
        for idx in range(3):
            plt.subplot(1, 3, idx + 1)
            plt.scatter(targets[idx], predictions[idx], alpha=0.6)
            
            # Add diagonal line
            min_val = min(targets[idx].min(), predictions[idx].min())
            max_val = max(targets[idx].max(), predictions[idx].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate R²
            correlation_matrix = np.corrcoef(targets[idx].flatten(), predictions[idx].flatten())
            r_squared = correlation_matrix[0, 1] ** 2
            
            plt.title(f'Model {idx} (R² = {r_squared:.3f})')
            plt.xlabel('Target')
            plt.ylabel('Prediction')
        
        plt.tight_layout()
        
        # Save plot
        save_directory = "./plots/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        plt.savefig(save_directory + "/predictions_vs_targets.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved predictions vs targets plot to {save_directory}")
    else:
        print("Prediction files not found. Run test_model_prediction_wandb first.")


def get_experiment_id(model_params, target_prop):
    """
    Get wandb experiment ID - equivalent to SigOpt experiment ID
    """
    f = open('inference/experiment_ids.json')
    settings_to_id = json.load(f)
    f.close()

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])

    if wandb_name in settings_to_id:
        return settings_to_id[wandb_name]
    else:
        raise ValueError('These model parameters have not been studied') 