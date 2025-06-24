def build_wandb_name(data_source,target_prop,struct_type,interpolation,model_type,contrastive_weight=1.0,training_fraction=1.0,training_seed=0,long_range=False):
    """
    Build wandb experiment name similar to sigopt naming convention
    This function creates consistent naming for wandb experiments
    """
    wandb_name = target_prop

    if data_source == "data/":
        wandb_name += "_" 
        wandb_name += "htvs_data"

    elif data_source == "pretrain_data/":
        wandb_name += "_" 
        wandb_name += "pretrain_data"
        
    elif data_source == "data_per_site/":
        wandb_name += "_" 
        wandb_name += "data_per_site"

    elif data_source == "data_per_site/":
        wandb_name += "_" 
        wandb_name += "data_per_site"

    wandb_name += "_" 
    wandb_name += struct_type

    if interpolation:
        wandb_name += "_" 
        wandb_name += "interpolation"

    wandb_name = wandb_name + "_" + model_type

    if long_range:
        wandb_name += "_" 
        wandb_name += "Long_Range"

    if contrastive_weight != 1.0:
        wandb_name += "_" 
        wandb_name += "ContrastiveWeight"
        wandb_name += str(contrastive_weight)

    if training_fraction != 1.0:
        wandb_name += "_" 
        wandb_name += "TrainingFraction"
        wandb_name += str(training_fraction)

    if training_seed != 0:
        wandb_name += "_" 
        wandb_name += "TrainingSeed"
        wandb_name += str(training_seed)

    return wandb_name 