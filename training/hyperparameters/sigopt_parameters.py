# Original SigOpt functions (commented out)
# def get_cgcnn_hyperparameter_range():
#     parameters=[
#         # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
#         dict(name="batch_size", bounds=dict(min=4, max=16), type="int"),
#         dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
#         dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"),
#         dict(name="atom_fea_len", bounds=dict(min=32, max=256), type="int"), 
#         dict(name="n_conv", bounds=dict(min=2, max=5), type="int"), 
#         dict(name="h_fea_len", bounds=dict(min=32, max=256), type="int"), 
#         dict(name="n_h", bounds=dict(min=1, max=4), type="int")
#     ]
#     return parameters

# def get_painn_hyperparameter_range():
#     parameters=[
#         # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
#         dict(name="batch_size", bounds=dict(min=4, max=16), type="int"),
#         dict(name="log_lr", bounds=dict(min=-5, max=-3), type="int"), 
#         dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"), 
#         dict(name="log2_feat_dim", bounds=dict(min=5, max=9), type="int"), 
#         dict(name="activation", categorical_values=["swish", "learnable_swish", "ReLU", "LeakyReLU"], type="categorical"),
#         dict(name="num_conv", bounds=dict(min=1, max=6), type="int"), 
#      ]
#     return parameters

# def get_e3nn_hyperparameter_range():
#     parameters=[
#         # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
#         dict(name="batch_size", bounds=dict(min=4, max=12), type="int"),
#         dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
#         dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"), 
#         dict(name="len_embedding_feature_vector", grid=[32, 64, 128], type="int"),
#         dict(name="num_hidden_feature", grid=[32, 64, 128], type="int"), 
#         dict(name="num_hidden_layer", bounds=dict(min=0, max=2), type="int"), 
#         dict(name="multiplicity_irreps", grid=[16, 32, 64], type="int"),
#         dict(name="num_conv", bounds=dict(min=1, max=4), type="int"), 
#         dict(name="num_radical_basis", grid=[5, 10, 20], type="int"),
#         dict(name="num_radial_neurons", grid=[32, 64, 128], type="int"), 
#     ]
#     return parameters

# Wandb equivalents (active)
def get_cgcnn_hyperparameter_range():
    """Get wandb sweep configuration for CGCNN model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'values': [4, 8, 12, 16]
            },
            'log_lr': {
                'min': -5,
                'max': -2,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'atom_fea_len': {
                'min': 32,
                'max': 256,
                'distribution': 'int_uniform'
            },
            'n_conv': {
                'min': 2,
                'max': 5,
                'distribution': 'int_uniform'
            },
            'h_fea_len': {
                'min': 32,
                'max': 256,
                'distribution': 'int_uniform'
            },
            'n_h': {
                'min': 1,
                'max': 4,
                'distribution': 'int_uniform'
            }
        }
    }
    return sweep_config


def get_painn_hyperparameter_range():
    """Get wandb sweep configuration for Painn model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'values': [4, 8, 12, 16]
            },
            'log_lr': {
                'min': -5,
                'max': -3,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'log2_feat_dim': {
                'min': 5,
                'max': 9,
                'distribution': 'int_uniform'
            },
            'activation': {
                'values': ['swish', 'learnable_swish', 'ReLU', 'LeakyReLU']
            },
            'num_conv': {
                'min': 1,
                'max': 6,
                'distribution': 'int_uniform'
            }
        }
    }
    return sweep_config


def get_e3nn_hyperparameter_range():
    """Get wandb sweep configuration for e3nn model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'values': [4, 6, 8, 10, 12]
            },
            'log_lr': {
                'min': -5,
                'max': -2,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'len_embedding_feature_vector': {
                'values': [32, 64, 128]
            },
            'num_hidden_feature': {
                'values': [32, 64, 128]
            },
            'num_hidden_layer': {
                'min': 0,
                'max': 2,
                'distribution': 'int_uniform'
            },
            'multiplicity_irreps': {
                'values': [16, 32, 64]
            },
            'num_conv': {
                'min': 1,
                'max': 4,
                'distribution': 'int_uniform'
            },
            'num_radical_basis': {
                'values': [5, 10, 20]
            },
            'num_radial_neurons': {
                'values': [32, 64, 128]
            }
        }
    }
    return sweep_config


def convert_hyperparameters(hyperparameters):
    """Convert wandb hyperparameters to the format expected by the training code"""
    # Convert log_lr to actual learning rate but keep both
    if 'log_lr' in hyperparameters:
        hyperparameters['lr'] = 10 ** hyperparameters['log_lr']
        # Keep log_lr as well since trainer expects it
    
    # Convert log2_feat_dim to feat_dim for Painn
    if 'log2_feat_dim' in hyperparameters:
        hyperparameters['feat_dim'] = 2 ** hyperparameters['log2_feat_dim']
        # Keep log2_feat_dim as well since trainer might expect it
    
    return hyperparameters


# def get_retrain_hyperparameter_range():
#     parameters=[
#         # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
#         dict(name="batch_size", bounds=dict(min=4, max=16), type="int"),
#         dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
#         dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"), 
#     ]
#     return parameters