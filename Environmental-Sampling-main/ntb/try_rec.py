#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# # Preparation stuff

# ## Connect to Drive

# In[ ]:





# In[2]:


connect_to_drive = False


# In[3]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[4]:


if connect_to_drive:
    #Install FS code
    get_ipython().system('pip install  --upgrade --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[5]:


#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import csv
import torch


# ## Define paths

# In[6]:


#every path should start from the project folder:
project_folder = "../"
if connect_to_drive:
    project_folder = "/content/gdrive/Shareddrives/<SharedDriveName>" #Name of SharedDrive folder
    #project_folder = "/content/gdrive/MyDrive/<MyDriveName>" #Name of MyDrive folder

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
img_folder = os.path.join(out_folder,"img")


# ## Import own code

# In[7]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory
# from src import ??? as additional_module
import easy_rec as additional_module #REMOVE THIS LINE IF IMPORTING OWN ADDITIONAL MODULE

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[8]:


cfg = easy_exp.cfg.load_configuration("config_rec")


# In[9]:


#cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

data_params = cfg["data_params"].copy()
data_params["data_folder"] = raw_data_folder

data, maps = easy_rec.data_generation_utils.preprocess_dataset(**data_params)


# In[10]:


# #Save user and item mappings
# # TODO: check
# with open(os.path.join(processed_data_folder,"user_map.csv"), "w") as f_user:
#     w = csv.writer(f_user)
#     w.writerows(maps['uid'].items())

# with open(os.path.join(processed_data_folder,"item_map.csv"), "w") as f_item:
#     w = csv.writer(f_item)
#     w.writerows(maps['sid'].items())


# In[11]:


datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**data_params["dataset_params"])


# In[12]:


collator_params = cfg["data_params"]["collator_params"].copy()
collator_params["num_items"] = np.max(list(maps["sid"].values()))


# In[13]:


# app = collator_params.get("negatives_distribution",None)
# if app is not None:
#     if app == "popularity":
#         collator_params["negatives_distribution"] = easy_rec.data_generation_utils.get_popularity_items(datasets["train"], collator_params["num_items"])
#     elif app not in ["uniform","dynamic"]:
#         raise ValueError("Invalid negatives distribution")


# In[14]:


collators = easy_rec.rec_torch.prepare_rec_collators(**collator_params)


# In[15]:


loader_params = cfg["model"]["loader_params"].copy()
loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, **loader_params, collate_fn=collators)


# In[16]:


rec_model_params = cfg["model"]["rec_model"].copy()
rec_model_params["num_items"] = np.max(list(maps["sid"].values()))
rec_model_params["num_users"] = np.max(list(maps["uid"].values()))
rec_model_params["lookback"] = data_params["collator_params"]["lookback"]


# In[17]:


main_module = easy_rec.rec_torch.create_rec_model(**rec_model_params)#, graph=easy_rec.data_generation_utils.get_graph_representation(data["train_sid"]))


# In[18]:


exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


# In[19]:


# # Find "original" implementation:
# # ...

# keys_to_change = {"model.rec_model.seed": 42}
# orig_cfg = deepcopy(cfg)
# for k,v in keys_to_change.items():
#     orig_cfg[k] = 42

# orig_exp_found, orig_experiment_id = easy_exp.exp.get_experiment_id(orig_cfg)
# print("Experiment already found:", orig_exp_found, "----> The experiment id is:", orig_experiment_id)

# Caricare modello originale (last o best) e fare predizione...
# Mettere la predizione dentro metrica RLS... --> prossime celle


# In[20]:


#if exp_found: exit() #TODO: make the notebook/script stop here if the experiment is already found


# In[21]:


model_params = cfg["model"].copy()

trainer_params = easy_torch.preparation.prepare_experiment_id(model_params["trainer_params"], experiment_id)

#dynamic_negatives_index = [i for i, x in enumerate(trainer_params["callbacks"]) if "DynamicNegatives" in x][0]
#trainer_params["callbacks"][dynamic_negatives_index]["DynamicNegatives"]["dataloader"] = loaders["train"]

# Prepare callbacks and logger using the prepared trainer_params
trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params, additional_module.callbacks)
trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)

# Prepare the trainer using the prepared trainer_params
trainer = easy_torch.preparation.prepare_trainer(**trainer_params)

model_params["loss"] = easy_torch.preparation.prepare_loss(model_params["loss"], additional_module.losses)

# Prepare the optimizer using configuration from cfg
model_params["optimizer"] = easy_torch.preparation.prepare_optimizer(**model_params["optimizer"])

# Prepare the metrics using configuration from cfg
# num_negatives = {split_name:[x] for split_name,x in data_params["collator_params"]["num_negatives"].items()}
# num_negatives["val"] += num_negatives["test"] #cause using test as val just to get metrics
# model_params["metrics"] = additional_module.metrics.prepare_rank_corrections(model_params["metrics"], num_negatives = num_negatives, num_items = rec_model_params["num_items"])
model_params["metrics"] = easy_torch.preparation.prepare_metrics(model_params["metrics"], additional_module.metrics)

# Create the model using main_module, loss, and optimizer
model = easy_torch.process.create_model(main_module, **model_params)


# In[22]:


# Prepare the emission tracker using configuration from cfg
#tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)


# In[23]:


# Prepare the flops profiler using configuration from cfg
#profiler = easy_torch.preparation.prepare_flops_profiler(model=model, **cfg["model"]["flops_profiler"], experiment_id=experiment_id)


# ### Train

# In[24]:


#easy_torch.process.test_model(trainer, model, loaders, test_key=["train","val","test"]) #, tracker=tracker, profiler=profiler)


# In[25]:


# Train the model using the prepared trainer, model, and data loaders
import time
start = time.time()
easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"]) #tracker=tracker, profiler=profiler, 
print("Elapsed time:", time.time()-start)


# In[26]:


easy_torch.process.test_model(trainer, model, loaders) #, tracker=tracker, profiler=profiler)


# In[27]:


# Save experiment and print the current configuration
#save_experiment_and_print_config(cfg)
easy_exp.exp.save_experiment(cfg)

# Print completion message
print("Execution completed.")
print("######################################################################")
print()

