#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[1]:


connect_to_drive = False


# In[2]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[3]:


if connect_to_drive:
    #Install FS code
    get_ipython().system('pip install  --upgrade --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[4]:


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

# In[5]:


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

# In[6]:


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

# In[7]:


cfg = easy_exp.cfg.load_configuration("config_rec")


# In[8]:


#cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

data_params = cfg["data_params"].copy()
data_params["data_folder"] = raw_data_folder

data, maps = easy_rec.data_generation_utils.preprocess_dataset(**data_params)


# In[9]:


item_infos = easy_rec.data_generation_utils.load_item_info('../data/raw', "ml-100k")
print(len(item_infos))
print(item_infos.keys())
print(item_infos[['movie_id', 'movie_title']].head(3))


# In[10]:


data['train_sid']


# In[11]:


item_infos['movie_title'].values


# In[12]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


# In[13]:


import pandas as pd
# Get all movie titles
titles = item_infos['movie_title'].tolist()

# Compute embeddings
print("Computing embeddings...")
embeddings = model.encode(titles, show_progress_bar=True)

# Create a DataFrame with movie IDs, titles, and their embeddings
embeddings_df = pd.DataFrame({
    'movie_id': item_infos['movie_id'],
    'title': item_infos['movie_title'],
    'embedding': list(embeddings)
})

print("\nShape of embeddings:", embeddings.shape)
print("\nFirst movie title and its embedding shape:")
print(f"Title: {embeddings_df['title'].iloc[0]}")
print(f"Embedding shape: {len(embeddings_df['embedding'].iloc[0])}")


# In[14]:


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between one movie and all others in list of lists data['train_sid']
# for each user save the corresponding movie title : movie similar with a cosine similarity threshold of 0.5

d = {}
counter = 1
for user_list in data['train_sid']:
    user_key = str(counter)
    d[user_key] = {}
    counter+=1

    for movie_id in range(len(user_list)):

        movie_embedding = embeddings_df[embeddings_df['movie_id'] == user_list[movie_id]]['embedding'].values[0]

        similar_movies = []
        
        for j in range(movie_id, len(user_list)): 
            comparison_movie_id = user_list[j]
            comparison_embedding = embeddings_df.loc[embeddings_df['movie_id'] == comparison_movie_id, 'embedding'].values[0]
            
            similarity = cosine_similarity([movie_embedding], [comparison_embedding])[0][0]
            
            if similarity > 0.9 and similarity < 0.98:  # so we avoid to save the same title
                movie_title = embeddings_df.loc[embeddings_df['movie_id'] == comparison_movie_id, 'title'].values[0]
                similar_movies.append(movie_title)
        if similar_movies != []:
            movie_title = embeddings_df.loc[embeddings_df['movie_id'] == user_list[movie_id], 'title'].values[0]
            d[user_key][movie_title] = similar_movies

# save the dictionary into a json file
import json
with open('similar_movies_09.json', 'w') as f:
    json.dump(d, f)




# In[15]:




