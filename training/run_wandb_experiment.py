import json
import sys
import pandas as pd
import argparse
import pickle as pkl
import torch
import numpy as np
import random
import shutil
import os
import wandb
# Original SigOpt imports (commented out)
# import sigopt
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from processing.create_model.create_model import create_model
from training.hyperparameters.wandb_parameters import *
from training.model_training.trainer import *
# Original SigOpt utils (commented out)
# from training.sigopt_utils import build_sigopt_name
from training.wandb_utils import build_wandb_name
from training.evaluate import *


def run_wandb_experiment(struct_type,model_type,gpu_num,experiment_id=None,parallel_band=1,obs_budget=50,training_fraction=1.0,data_name="data/",target_prop="dft_e_hull",interpolation=False,contrastive_weight=1.0,training_seed=0,nickname=""):
    """Run wandb hyperparameter optimization experiment"""
    
    # Original SigOpt code (commented out)
    # if experiment_id == None:
    #     sigopt_settings={"parallel_band": parallel_band,"obs_budget": obs_budget}

    if data_name == "data/":
        training_data = pd.read_json(data_name + 'training_set.json')
        training_data = training_data.sample(frac=training_fraction,replace=False,random_state=training_seed)
        validation_data = pd.read_json(data_name + 'validation_set.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')
        if not interpolation:
            training_data = pd.concat((training_data,edge_data))
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

    # Original SigOpt code (commented out)
    # conn = sigopt.Connection(driver="lite")
    # sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    # 
    # if experiment_id == None:
    #     experiment = create_sigopt_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,sigopt_settings,conn)
    #     print("Created a new SigOpt experiment '" + sigopt_name + "' with ID: " + str(experiment.id))
    # else:
    #     experiment = conn.experiments(experiment_id).fetch()
    #     print("Continuing a prior SigOpt experiment '" + sigopt_name + "' with ID: " + str(experiment.id))
    # 
    # while experiment.progress.observation_count < experiment.observation_budget:
    #     print('\n========================\nSigopt experiment count #', experiment.progress.observation_count)
    #     suggestion = conn.experiments(experiment.id).suggestions().create()
    #     hyperparameters = suggestion.assignments
    # 
    #     value = sigopt_evaluate_model(data_name,suggestion.assignments,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment.id,experiment.progress.observation_count,gpu_num,nickname)    
    #     training_results = {"validation_loss": value}
    # 
    #     conn.experiments(experiment.id).observations().create(
    #         suggestion=suggestion.id,
    #         values=[{"name": "val_mae", "value": value}],
    #     )
    # 
    #     experiment = conn.experiments(experiment.id).fetch()
    #     observation_id = experiment.progress.observation_count - 1
    # 
    #     model_save_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment.id) + '/observ_' + str(observation_id)
    #     model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment_id) + '/' + nickname + '_tmp' + str(gpu_num)
    # 
    #     with open(model_tmp_dir + '/hyperparameters.json', 'w') as file:
    #         json.dump(hyperparameters, file)
    #     with open(model_tmp_dir + '/training_results.json', 'w') as file:
    #         json.dump(training_results, file)
    # 
    #     if not os.path.exists(model_save_dir):
    #         os.makedirs(model_save_dir)
    # 
    #     ### Copy contents of tmp file
    #     possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
    #                            "final_model.torch","final_model","final_model.pth.tar",
    #                            "log_human_read.csv","checkpoints/checkpoint-100.pth.tar",
    #                            'hyperparameters.json','training_results.json']
    #     for file_name in possible_file_names:
    #         if os.path.isfile(model_tmp_dir + "/" + file_name):
    #             if file_name == "checkpoints/checkpoint-100.pth.tar":
    #                 shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
    #             else:
    #                 shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
    #     
    #     ### Empty tmp file
    #     shutil.rmtree(model_tmp_dir)
    #     torch.cuda.empty_cache()

    # Wandb equivalent (active)
    # Create wandb sweep configuration
    sweep_config = create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget)
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    print(f"Created wandb sweep configuration for '{wandb_name}'")
    print(f"Model type: {model_type}")
    print(f"Structure type: {struct_type}")
    print(f"Project name from config: {sweep_config.get('project', 'NOT_FOUND')}")

    # Initialize sweep with dynamic project name
    project_name = sweep_config['project']
    print(f"Using project name: {project_name}")
    
    # Remove project from sweep_config to avoid conflicts
    sweep_config_without_project = sweep_config.copy()
    del sweep_config_without_project['project']
    
    sweep_id = wandb.sweep(sweep_config_without_project, project=project_name)
    print(f"Created wandb sweep with ID: {sweep_id}")
    print(f"Project: {project_name}")

    # Define the training function for the sweep
    def train_function():
        # Initialize wandb run
        wandb.init()
        
        # Get hyperparameters from wandb
        hyperparameters = dict(wandb.config)
        
        # Convert hyperparameters to expected format
        hyperparameters = convert_hyperparameters(hyperparameters)
        
        # Train model
        val_loss = wandb_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,sweep_id,obs_budget,gpu_num,nickname)
        
        # Log final validation loss
        wandb.log({"val_mae": val_loss})
        
        # Save model files permanently (equivalent to sigopt storage)
        run_id = wandb.run.id
        model_save_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(sweep_id) + '/observ_' + str(run_id)
        model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
        
        # Ensure the temporary directory exists
        if not os.path.exists(model_tmp_dir):
            os.makedirs(model_tmp_dir)
        
        # Save hyperparameters and training results
        training_results = {"validation_loss": val_loss}
        with open(model_tmp_dir + '/hyperparameters.json', 'w') as file:
            json.dump(hyperparameters, file)
        with open(model_tmp_dir + '/training_results.json', 'w') as file:
            json.dump(training_results, file)
        
        # Create permanent save directory
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        # Copy contents of tmp file to permanent location
        possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
                               "final_model.torch","final_model","final_model.pth.tar",
                               "log_human_read.csv","checkpoints/checkpoint-100.pth.tar",
                               'hyperparameters.json','training_results.json']
        for file_name in possible_file_names:
            if os.path.isfile(model_tmp_dir + "/" + file_name):
                if file_name == "checkpoints/checkpoint-100.pth.tar":
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
                else:
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
        
        # Clean up tmp directory
        shutil.rmtree(model_tmp_dir)
        torch.cuda.empty_cache()
    
    # Run the sweep
    wandb.agent(sweep_id, train_function, count=obs_budget)
    
    print(f"Completed wandb sweep with {obs_budget} observations")


def wandb_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment_id,observation_count,gpu_num,nickname):
    """Evaluate model for wandb experiment"""
    
    # Original SigOpt evaluation code (commented out)
    # device = "cuda:" + str(gpu_num)
    # train_data = processed_data[0]
    # validation_data = processed_data[1]
    # per_site = False
    # if "per_site" in target_prop:
    #     per_site = True
    # train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation,per_site=per_site)
    # train_eval_loader = None
    # if "e3nn" in model_type and "pretrain" not in data_name and "per_site" not in target_prop:
    #     train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
    #     val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
    # else:
    #     val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site)
    # model, normalizer = create_model(model_type,train_loader,interpolation,target_prop,hyperparameters=hyperparameters,per_site=per_site)
    # sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    # model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment_id) + '/' + nickname + '_tmp' + str(gpu_num)
    # if os.path.exists(model_tmp_dir):
    #     shutil.rmtree(model_tmp_dir)
    # os.makedirs(model_tmp_dir) 
    # best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader=train_eval_loader,contrastive_weight=contrastive_weight)
    # is_contrastive = False
    # if "contrastive" in model_type:
    #     is_contrastive = True
    # _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)
    # if model_type == "Painn":
    #     return best_loss
    # else:
    #     return best_loss[0]

    # Wandb equivalent (active)
    device = "cuda:" + str(gpu_num)
    
    train_data = processed_data[0]
    validation_data = processed_data[1]

    per_site = False
    if "per_site" in target_prop:
        per_site = True

    # Ensure hyperparameters is a dict and convert if needed
    if hyperparameters is None:
        hyperparameters = {}
    elif not isinstance(hyperparameters, dict):
        hyperparameters = dict(hyperparameters)
    
    # Convert hyperparameters to expected format
    hyperparameters = convert_hyperparameters(hyperparameters)

    train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation,per_site=per_site)
    train_eval_loader = None

    if "e3nn" in model_type and "pretrain" not in data_name and "per_site" not in target_prop:
        train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
        val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
    else:
        val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site)
    
    # Pass hyperparameters as positional argument
    model, normalizer = create_model(model_type, train_loader, interpolation, target_prop, hyperparameters, per_site=per_site)
    
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    # Get run ID safely
    run_id = "unknown"
    if wandb.run is not None:
        run_id = wandb.run.id
    
    model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
    if os.path.exists(model_tmp_dir):
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir) 

    best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader=train_eval_loader,contrastive_weight=contrastive_weight)
    
    is_contrastive = False
    if "contrastive" in model_type:
        is_contrastive = True
    _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)

    if model_type == "Painn":
        return best_loss
    else:
        return best_loss[0]


def create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget):
    """Create wandb sweep configuration"""
    
    # Original SigOpt experiment creation (commented out)
    # sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    # if model_type == "Painn":
    #     curr_parameters = get_painn_hyperparameter_range()
    # elif model_type == "CGCNN":
    #     curr_parameters = get_cgcnn_hyperparameter_range()
    # else:
    #     curr_parameters = get_e3nn_hyperparameter_range()
    # experiment = conn.experiments().create(
    #     name=sigopt_name, 
    #     parameters = curr_parameters,
    #     metrics=[dict(name="val_mae", objective="minimize", strategy="optimize")],
    #     observation_budget=sigopt_settings["obs_budget"], 
    #     parallel_bandwidth=sigopt_settings["parallel_band"],
    # )
    # return experiment

    # Wandb equivalent (active)
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    if model_type == "Painn":
        sweep_config = get_painn_hyperparameter_range()
    elif model_type == "CGCNN":
        sweep_config = get_cgcnn_hyperparameter_range()
    else:
        sweep_config = get_e3nn_hyperparameter_range()
    
    # Create project name based on model type and structure type
    # 6 different project names: CGCNN-unrelaxed, CGCNN-relaxed, e3nn-unrelaxed, e3nn-relaxed, Painn-unrelaxed, Painn-relaxed
    if struct_type in ["unrelaxed", "relaxed"]:
        project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
    else:
        # For other structure types, use a generic name
        project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
    
    # Add project and program info
    sweep_config.update({
        'project': project_name,
        'name': wandb_name
    })
    
    return sweep_config


def convert_hyperparameters(hyperparameters):
    """Convert wandb hyperparameters to the format expected by the model"""
    # This function should convert the hyperparameters from wandb format
    # to the format expected by the create_model function
    # You may need to adjust this based on your specific hyperparameter structure
    return hyperparameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for perovskite ordering GCNNs using wandb')
    parser.add_argument('--data_name', default = "data/", type=str, metavar='name',
                        help="the source of the data")
    parser.add_argument('--prop', default = "dft_e_hull", type=str, metavar='name',
                        help="the property to predict (default: dft_e_hull; other options: Op_band_center)")
    parser.add_argument('--struct_type', default = 'unrelaxed', type=str, metavar='struct_type',
                        help="using which structure representation (default: unrelaxed; other options: relaxed, M3Gnet_relaxed)")
    parser.add_argument('--interpolation', default = 'no', type=str, metavar='yes/no',
                        help="using interpolation (default: no; other options: yes)")
    parser.add_argument('--model', default = "CGCNN", type=str, metavar='model',
                        help="the neural network to use (default: CGCNN; other options: Painn, e3nn, e3nn_contrastive)")
    parser.add_argument('--contrastive_weight', default = 1.0, type=float, metavar='loss_parameters',
                        help="the weighting applied to the contrastive loss term (default: 1.0)")
    parser.add_argument('--training_fraction', default = 1.0, type=float, metavar='training_set',
                        help="fraction of the total training set used (default 1.0)")
    parser.add_argument('--training_seed', default = 0, type=int, metavar='training_set',
                        help="the random seed for selecting fraction of training set (default 0)")
    parser.add_argument('--gpu', default = 0, type=int, metavar='device',
                        help="the gpu to use (default: 0)")
    parser.add_argument('--nickname', default = "", type=str, metavar='device',
                        help="nickname for temporary folder")
    parser.add_argument('--budget', default = 50, type=int, metavar='wandb_props',
                        help="budget of wandb sweep (default: 50)")
    args = parser.parse_args()

    data_name = args.data_name
    target_prop = args.prop
    model_type = args.model
    gpu_num = args.gpu
    nickname = args.nickname
    struct_type = args.struct_type
    contrastive_weight = args.contrastive_weight
    training_fraction = args.training_fraction
    training_seed = args.training_seed
    obs_budget = args.budget
    
    if struct_type not in ["unrelaxed", "relaxed", "spud", "M3Gnet_relaxed"]:
        raise ValueError('struct type is not available')
    
    if args.interpolation == 'yes':
        interpolation = True
    elif args.interpolation == 'no':
        interpolation = False
    else:
        raise ValueError('interpolation needs to be yes or no')    
    
    run_wandb_experiment(struct_type,model_type,gpu_num,None,1,obs_budget,training_fraction,data_name,target_prop,interpolation,contrastive_weight,training_seed,nickname) 