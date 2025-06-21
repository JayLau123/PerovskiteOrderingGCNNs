#!/usr/bin/env python
# coding: utf-8

# **This notebook provides examples of how to train GCNNs on the training dataset and conduct hyperparameter optimization based on the loss on the validation set (function: run_sigopt_experiment).**
# 
# Parameters:
# - struct_type: the structure representation to use (options: unrelaxed, relaxed, M3Gnet_relaxed)
# - model_type: the model architechture to use (options: CGCNN, e3nn, Painn)
# - gpu_num: the GPU to use
# - obs_budget: the budget of hyperparameter optimization
# - training_fraction: if not trained on the entire training set, the fraction of the training set to use
# - training_seed: if not trained on the entire training set, the random seed for selecting this fraction

# In[1]:


from training.run_sigopt_experiment import run_sigopt_experiment


# # CGCNN

# In[2]:


run_sigopt_experiment(
    struct_type="unrelaxed",
    model_type="CGCNN",
    gpu_num=0,
    obs_budget=1,
    training_fraction=0.125,
    training_seed=0
)


# # e3nn

# In[3]:


run_sigopt_experiment(
    struct_type="relaxed",
    model_type="e3nn",
    gpu_num=0,
    obs_budget=1,
    training_fraction=0.125,
    training_seed=0
)


# # Painn

# In[4]:


run_sigopt_experiment(
    struct_type="unrelaxed",
    model_type="Painn",
    gpu_num=0,
    obs_budget=1,
    training_fraction=0.125,
    training_seed=0
)


# In[ ]:





# In[ ]:





# In[ ]:




