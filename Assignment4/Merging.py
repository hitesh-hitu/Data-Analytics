#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# loading the data
usage = pd.read_csv("user_usage.csv")
device = pd.read_csv("user_device.csv")


# In[3]:


usage.head()


# In[4]:


device.head()


# In[19]:


# inner merging which keeps only matched column values between left and right dataframes. 

m_inner = pd.merge(usage, device[['platform', 'device', 'use_id']], on = 'use_id')
m_inner.head()


# In[20]:


m_inner.tail()


# In[15]:


# left merge which keeps the left values, whereas keeps the matched with right values and all other values in right dataframe as NaN

m_left = pd.merge(usage, device[['platform', 'device', 'use_id']], how = 'left', on = 'use_id')
m_left.head()


# In[16]:


m_left.tail()


# In[17]:


# right merge which keeps the right values, whereas keeps the matched with left values and all other values in left dataframe as NaN

m_right = pd.merge(usage, device[['platform', 'device', 'use_id']], how = 'right', on = 'use_id')
m_right.head()


# In[18]:


m_right.tail()


# In[21]:


# outer merging will keep matched column values between left and right dataframes, and also keeps the cells which are not
# matched in both left and right dataframes with NaN values

m_outer = pd.merge(usage, device[['platform', 'device', 'use_id']], how = 'outer', on = 'use_id')
m_outer.head()


# In[22]:


m_outer.tail()


# In[ ]:




