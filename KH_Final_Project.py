#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Pre-Processing

# In[2]:


anime = pd.read_csv("Data/anime.csv")
anime


# In[3]:


for i in anime:
    print("{} : {} Null Values".format(i, anime[i].isna().sum().sum()))


# In[4]:


anime.dtypes


# In[5]:


# fill null values in 'genre' column with 'Unknown'
anime['genre'].fillna('Unknown', inplace=True)

# fill null values in 'type' column with 'Unknown'
anime['type'].fillna('Unknown', inplace=True)

# replace unknown values in episode column with NaN
anime["episodes"].replace("Unknown", np.nan, inplace=True)


# In[6]:


for i in anime:
    print("{} : {} Null Values".format(i, anime[i].isna().sum().sum()))


# In[7]:


def fill_null_with_genre_mode(df, column_name, group_by_column_name):
    groupby = df.groupby(group_by_column_name)[column_name]
    df[column_name] = groupby.apply(lambda x: x.fillna(method='ffill'))
    return df


# In[8]:


# fill null values in 'rating' column with the mode rating value for each genre
anime = fill_null_with_genre_mode(anime, 'rating', 'genre')


# In[9]:


for i in anime:
    print("{} : {} Null Values".format(i, anime[i].isna().sum().sum()))


# In[10]:


anime["rating"].fillna(anime["rating"].median(),inplace = True)


# In[11]:


for i in anime:
    print("{} : {} Null Values".format(i, anime[i].isna().sum().sum()))


# In[12]:


anime["episodes"].fillna(anime["episodes"].median(),inplace = True)


# In[13]:


anime


# In[14]:


pd.get_dummies(anime[["type"]]).head()


# In[15]:


# create dummy variables for genre and type columns
genre_dummies = pd.concat([anime["genre"].str.get_dummies(sep=",")])

type_dummies = pd.get_dummies(anime["type"])

# select columns to keep
columns_to_keep = ["rating", "members", "episodes"]

# create anime_features dataframe by concatenating the dummy variables and selected columns
anime_features = pd.concat([genre_dummies, type_dummies, anime[columns_to_keep]], axis=1)


# In[16]:


anime["name"] = anime["name"].map(lambda name: re.sub('[^A-Za-z0-9]+', ' ', name))


# In[17]:


anime_features


# In[18]:


anime_features.dtypes


# In[ ]:





# # KNN   

# In[19]:


scaledData = StandardScaler()
anime_features = scaledData.fit_transform(anime_features)


# In[20]:


nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(anime_features)


# In[21]:


distances, indices = nbrs.kneighbors(anime_features)


# In[22]:


# Calculate silhouette score
silhouette_avg = silhouette_score(anime_features, labels=indices[:, 1])
print("The average silhouette_score is :", silhouette_avg)


# In[ ]:





# In[ ]:





# In[23]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def anime_recommendation(anime_name, anime_data, anime_features):
    # Scale the features
    scaled_data = StandardScaler()
    anime_features_scaled = scaled_data.fit_transform(anime_features)

    # Fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(anime_features_scaled)
    
    # Find the distances and indices of the nearest neighbors
    distances, indices = nbrs.kneighbors(anime_features_scaled)

    # Find the index of the input anime in the anime_data DataFrame
    anime_index = anime_data[anime_data['name'] == anime_name].index[0]

    # Get the indices of the 5 closest anime (excluding the input anime)
    top_5_indices = indices[anime_index][1:]

    # Return the top 5 recommended anime names
    return anime_data.iloc[top_5_indices]['name'].values


# # Results

# In[24]:


anime_name = "Fairy Tail"
recommended_anime = anime_recommendation(anime_name, anime, anime_features)
print(recommended_anime)


# In[25]:


anime_name = "Bleach"
recommended_anime = anime_recommendation(anime_name, anime, anime_features)
print(recommended_anime)


# In[26]:


anime_name = "Noragami"
recommended_anime = anime_recommendation(anime_name, anime, anime_features)
print(recommended_anime)


# # End
