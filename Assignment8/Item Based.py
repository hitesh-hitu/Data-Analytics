#!/usr/bin/env python
# coding: utf-8

# In[1]:


### ITEM BASED FILTERING
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('ratings.csv',
              names=['user_id','movie_id','rating','timestamp'])
df.head()


# In[3]:


movie_titles=pd.read_csv('movies.csv', 
                         names=['movie_id','title','genres'])
movie_titles.head()


# In[4]:


df=pd.merge(df,movie_titles,on='movie_id')
df=df[['user_id','movie_id','rating','timestamp','title']]
df.head()


# In[5]:


df.describe()


# In[6]:


#each movie with it's average rating along with how many people have rated it.
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['number_of_ratings']=df.groupby('title')['rating'].count()
ratings.head()


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ratings['rating'].hist(bins=50)

#We can see that most of the movies are rated between 2.5 and 4


# In[8]:


ratings['number_of_ratings'].hist(bins=60)


# In[9]:


#Check the relationship between the rating of the movie and no. of ratings
#Seaborn's jointplot() helps us visualize this
import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
#clearly there is a positive rel. b/w average rating and no. of ratings
#the more the ratings a movie gets, higher the average rating it gets
#This is important to note especially when choosing the threshold for the number of ratings per movie.


# In[9]:


#create a matrix with movie_titles as columns and user_id as rows(index)
#use pandas' pivot_table() to create it

movie_matrix=df.pivot_table(index='user_id',columns='title',values='rating')
movie_matrix.head()


# In[10]:


ratings.sort_values('number_of_ratings', ascending=False).head(10)


# In[11]:


'''
Assume that a user has watched 2 movies A and B,we have to recommend movies to this user based 
on this watching history. Hence we should look for movies that are similar to A and B.
We do this by computing the correlation between these 2 movies' ratings and the ratings
of the rest of the movies.
Assume A to be "Matrix, The (1999)" and B to be "Pulp Fiction (1994)"
'''
Matrix_user_rating = movie_matrix['Matrix, The (1999)']
Pulp_user_rating = movie_matrix['Pulp Fiction (1994)']
Matrix_user_rating.head()


# In[12]:


Pulp_user_rating.head()


# In[13]:


#use corrwith() to find correlation between 2 dataframes,columns etc.
similar_to_Matrix = movie_matrix.corrwith(Matrix_user_rating)
similar_to_Matrix.head(50)


# In[14]:


similar_to_Pulp=movie_matrix.corrwith(Pulp_user_rating)
similar_to_Pulp.head(50)


# In[15]:


#Clearly there are a lot of null values, hence we drop these rows.
corr_Matrix=pd.DataFrame(similar_to_Matrix,columns=['Correlation'])
corr_Matrix.dropna(inplace=True)
corr_Matrix.head()


# In[16]:


corr_Pulp=pd.DataFrame(similar_to_Pulp,columns=['Correlation'])
corr_Pulp.dropna(inplace=True)
corr_Pulp.head()


# In[17]:


#In the above, we didn't set a threshold to the number of ratings for the movie
# i.e only one person could've rated it high, and that doesn't make sense
#Now we set the threshold
corr_Matrix = pd.merge(corr_Matrix,ratings,on="title")
#corr_Matrix = corr_Matrix[['title','Correlation','number_of_ratings']]
corr_Matrix.head()


# In[18]:


corr_Pulp = pd.merge(corr_Pulp,ratings,on="title")
#corr_Matrix = corr_Matrix[['title','Correlation','number_of_ratings']]
corr_Pulp.head()


# In[19]:


corr_Matrix[corr_Matrix['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)


# In[20]:


corr_Pulp[corr_Pulp['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)


# In[ ]:


#Recommend anything apart from the first one

