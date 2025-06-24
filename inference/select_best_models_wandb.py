import os
import shutil
import json
# import sigopt  # Original SigOpt import (commented out)
import math
import torch
import random
import numpy as np
import pandas as pd
import wandb  # Wandb import (active)
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from training.evaluate import evaluate_model
from processing.interpolation.Interpolation import *
from training.loss import contrastive_loss
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from processing.create_model.create_model import create_model

saved_models_path = "./saved_models/"

def get_experiment_id(model_params, target_prop):
    """
    Get wandb experiment ID - equivalent to SigOpt experiment ID
    """
    f = open('inference/experiment_ids.json')
    settings_to_id = json.load(f)
    f.close()

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    print(wandb_name)

    if wandb_name in settings_to_id:
        return settings_to_id[wandb_name]
    else:
        raise ValueError('These model parameters have not been studied')

def load_model(gpu_num, train_loader, target_prop, model_params, folder_idx, job_idx, per_site):
    """
    Load model - equivalent to SigOpt version but uses wandb naming
    """
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    exp_id = get_experiment_id(model_params, target_prop)
    directory = saved_models_path + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)+"/" + "observ_" + str(folder_idx)
    
    # Original SigOpt connection (commented out)
    # conn = sigopt.Connection(driver="lite")
    # all_observations = conn.experiments(exp_id).observations().fetch()
    # assignments = all_observations.data[job_idx].assignments

    if model_params["model_type"] == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        with open(directory + "/hyperparameters.json") as json_file:
            assignments = json.load(json_file)
        model, normalizer = create_model(model_params["model_type"],train_loader,model_params["interpolation"],target_prop,hyperparameters=assignments,per_site=per_site)
        model.to(device)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    
    return model, normalizer


def reverify_wandb_models(model_params, gpu_num, target_prop="dft_e_hull"):
    """
    Reverify wandb models - equivalent to reverify_sigopt_models
    """
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False

    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    
    interpolation = model_params["interpolation"]
    model_type = model_params["model_type"]    
    data_name = model_params["data"]
    struct_type = model_params["struct_type"]

    if data_name == "data/":
        training_data = pd.read_json(data_name + 'training_set.json')
        training_data = training_data.sample(frac=model_params["training_fraction"],replace=False,random_state=0)
        validation_data = pd.read_json(data_name + 'validation_set.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')

        if not interpolation:
            training_data = pd.concat((training_data,edge_data))
            
    # elif data_name == "data_per_site/":
    #     training_data = pd.read_json(data_name + 'training_set.json')
    #     training_data = training_data.sample(frac=model_params["training_fraction"],replace=False,random_state=0)
    #     validation_data = pd.read_json(data_name + 'validation_set.json')
    #     edge_data = pd.read_json(data_name + 'edge_dataset.json')

    #     if not interpolation:
    #         training_data = pd.concat((training_data,edge_data))

    # elif data_name == "pretrain_data/":

    #     training_data = pd.read_json(data_name + 'training_set.json')
    #     validation_data = pd.read_json(data_name + 'validation_set.json')

    else:
        print("Specified Data Directory Does Not Exist!")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print("Loaded data")

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,struct_type)
        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)
        processed_data.append(dataset)

    print("Completed data processing")
    
    train_data = processed_data[0]
    validation_data = processed_data[1]
    
    per_site = False
    if "per_site" in target_prop:
        per_site = True

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])
    val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])       

    reverify_wandb_models_results = pd.DataFrame(columns=['reverified_loss'])

    # Original SigOpt connection (commented out)
    # conn = sigopt.Connection(driver="lite")
    # all_observations = conn.experiments(exp_id).observations().fetch()

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    exp_id = get_experiment_id(model_params, target_prop)
    parent_directory = saved_models_path + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)
    num_folders = len([i for i in os.listdir(parent_directory) if os.path.isdir(parent_directory + "/" + i)])- 1

    for folder_idx in range(num_folders):        
        job_idx = num_folders - folder_idx - 1
        print('Reverifying wandb model #' + str(folder_idx))

        # Original SigOpt loss retrieval (commented out)
        # sigopt_loss = all_observations.data[job_idx].value
        # hyperparameters = all_observations.data[job_idx].assignments

        directory = saved_models_path + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)+"/" + "observ_" + str(folder_idx)
        # with open(directory + '/hyperparameters.json', 'w') as file:
        #     json.dump(hyperparameters, file)

        model, normalizer = load_model(gpu_num, train_loader, target_prop, model_params, folder_idx, job_idx,per_site=per_site)
        
        if "contrastive" in model_type:
            loss_fn = contrastive_loss 
            is_contrast = True
        else:
            loss_fn = torch.nn.L1Loss()
            is_contrast = False
            
        _, _, best_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrast,contrastive_weight=model_params["contrastive_weight"])

        if model_type == "Painn":
            reverified_loss = best_loss
        else:
            reverified_loss = best_loss[0]

        new_row = pd.DataFrame([[reverified_loss]], columns=['reverified_loss'])

        reverify_wandb_models_results = pd.concat([
            reverify_wandb_models_results,
            new_row
        ], ignore_index=True)

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    exp_id = get_experiment_id(model_params, target_prop)
    save_directory = saved_models_path + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)
    reverify_wandb_models_results.to_csv(save_directory + "/reverify_wandb_models_results.csv")


def keep_the_best_few_models(model_params, num_best_models=3, target_prop="dft_e_hull"):
    """
    Keep the best few models - equivalent to SigOpt version but uses wandb naming
    """
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    exp_id = get_experiment_id(model_params, target_prop)
    old_directory_prefix = saved_models_path + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)
    new_directory_prefix = "./best_models/" + model_params["model_type"] + "/"+ wandb_name + "/" +str(exp_id)

    reverify_wandb_models_results = pd.read_csv(old_directory_prefix + '/reverify_wandb_models_results.csv', index_col=0)

    # Original SigOpt verification (commented out)
    # for index, row in reverify_wandb_models_results.iterrows():
    #     if not math.isclose(row['sigopt_loss'], row['reverified_loss'], rel_tol=1e-2):
    #         print("========\nSigopt model #" + str(index) + " has non-matching losses\n========")

    if not os.path.exists(new_directory_prefix):
        os.makedirs(new_directory_prefix)

    for i in range(num_best_models):
        folder_idx = reverify_wandb_models_results.sort_values(by=['reverified_loss']).index[i]
        old_directory = old_directory_prefix + "/observ_" + str(folder_idx)
        new_directory = new_directory_prefix + "/best_" + str(i)
        
        if os.path.exists(old_directory):
            shutil.copytree(old_directory, new_directory)
            print(f"Copied model {folder_idx} to best_{i}")
        else:
            print(f"Warning: Directory {old_directory} does not exist")

    print(f"Kept the best {num_best_models} models in {new_directory_prefix}") 