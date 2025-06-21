#!/usr/bin/env python
# coding: utf-8

# **This notebook provides examples of how to verify the performance of GCNNs on the validation set (function: reverify_sigopt_models), select the top-performing models accordingly (function: keep_the_best_few_models), compute the prediction on the test and holdout sets (function: get_all_model_predictions), and extract the latent embeddings of CGCNN and e3nn after all message passing and graph convolution layers (function: get_all_embeddings).**
# 
# Parameters:
# - struct_type: the structure representation to use (options: unrelaxed, relaxed, M3Gnet_relaxed)
# - model_type: the model architechture to use (options: CGCNN, e3nn, Painn)
# - gpu_num: the GPU to use
# - training_fraction: if not trained on the entire training set, the fraction of the training set to use
# - num_best_models: the number of top-performing models to use

# In[1]:


from inference.select_best_models import reverify_sigopt_models, keep_the_best_few_models
from inference.test_model_prediction import get_all_model_predictions
from inference.embedding_extraction import get_all_embeddings


# # CGCNN

# In[2]:


reverify_sigopt_models(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "CGCNN",
        "training_fraction":1.0,
    },
    gpu_num=0
)


# In[3]:


keep_the_best_few_models(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "CGCNN",
        "training_fraction":1.0,
    },
    num_best_models=3
)


# In[4]:


get_all_model_predictions(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "CGCNN",
        "training_fraction":1.0,
    },
    gpu_num=0,
    num_best_models=3
)


# In[5]:


get_all_embeddings(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "CGCNN",
        "training_fraction":1.0,
    },
    gpu_num=0,
    num_best_models=3
)


# # e3nn

# In[6]:


reverify_sigopt_models(
    model_params={
        "struct_type": "relaxed",
        "model_type": "e3nn",
        "training_fraction":0.5,
    },
    gpu_num=0
)


# In[7]:


keep_the_best_few_models(
    model_params={
        "struct_type": "relaxed",
        "model_type": "e3nn",
        "training_fraction":0.5,
    },
    num_best_models=3
)


# In[8]:


get_all_model_predictions(
    model_params={
        "struct_type": "relaxed",
        "model_type": "e3nn",
        "training_fraction":0.5,
    },
    gpu_num=0,
    num_best_models=3
)


# In[9]:


get_all_embeddings(
    model_params={
        "struct_type": "relaxed",
        "model_type": "e3nn",
        "training_fraction":0.5,
    },
    gpu_num=0,
    num_best_models=3
)


# # Painn

# In[10]:


reverify_sigopt_models(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "Painn",
        "training_fraction":1.0,
    },
    gpu_num=0
)


# In[11]:


keep_the_best_few_models(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "Painn",
        "training_fraction":1.0,
    },
    num_best_models=3
)


# In[12]:


get_all_model_predictions(
    model_params={
        "struct_type": "unrelaxed",
        "model_type": "Painn",
        "training_fraction":1.0,
    },
    gpu_num=0,
    num_best_models=3
)


# In[ ]:





# In[ ]:





# In[ ]:




