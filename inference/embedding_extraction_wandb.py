import torch
import numpy as np
import pandas as pd
import json
import os
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from processing.create_model.create_model import create_model

def extract_embeddings(model_params, gpu_num, target_prop="dft_e_hull"):
    """
    Extract embeddings using wandb models - equivalent to SigOpt version
    """
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
    else:
        print("Specified Data Directory Does Not Exist!")

    torch.manual_seed(0)
    print("Loaded data")

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,struct_type)
        processed_data.append(dataset)

    print("Completed data processing")
    
    train_data = processed_data[0]
    validation_data = processed_data[1]
    
    per_site = False
    if "per_site" in target_prop:
        per_site = True

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation,per_site=per_site)
    val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site)

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    exp_id = get_experiment_id(model_params, target_prop)
    directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" +str(exp_id) + "/" + "best_"

    embeddings_list = []
    targets_list = []
    
    for idx in range(3):
        if model_params["model_type"] == "Painn":
            model = torch.load(directory + str(idx) + "/best_model", map_location=device)
            normalizer = None
        else:
            with open(directory + str(idx) + "/hyperparameters.json") as json_file:
                assignments = json.load(json_file)
            model, normalizer = create_model(model_params["model_type"],train_loader,model_params["interpolation"],target_prop,hyperparameters=assignments,per_site=per_site)
            model.to(device)
            model.load_state_dict(torch.load(directory + str(idx) + "/best_model.torch", map_location=device)['state'])
        
        model.eval()
        
        embeddings = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if model_params["model_type"] == "Painn":
                    batch_embeddings = model.get_embeddings(batch)
                else:
                    batch_embeddings = model(batch, return_embeddings=True)
                
                batch_targets = batch[1]
                
                embeddings.append(batch_embeddings.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        embeddings_list.append(embeddings)
        targets_list.append(targets)
    
    # Save embeddings
    save_directory = "./embeddings/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    np.save(save_directory + "/embeddings.npy", np.array(embeddings_list))
    np.save(save_directory + "/targets.npy", np.array(targets_list))
    
    print(f"Saved embeddings to {save_directory}")


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