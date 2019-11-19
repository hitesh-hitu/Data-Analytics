#!/usr/bin/env python
# coding: utf-8

# In[2]:


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt  # major lib for smoothening
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.seterr(divide = 'ignore') 


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

dataset = pd.read_csv('C:/Users/Hitesh Kumar/Videos/Sem 5/DA/Assignments/AirPassengers.csv',parse_dates=['Month'],
                     index_col='Month',date_parser=dateparse)

ds = dataset['#Passengers']
ds.head()


ds.plot(kind="line",figsize=(10,5))

my_data = np.log(ds)


# In[7]:


fit1 = Holt(my_data).fit(smoothing_level=0.8, smoothing_slope=0.6, optimized=False)
fit1_1 = fit1.forecast(len(my_data)).rename("Holt's linear trend")

my_data.plot(kind="line",figsize=(10,5))
fit1_1.plot(kind="line",figsize=(10,5),color='orange')


# In[10]:


fit2 = Holt(my_data).fit(smoothing_level=0.2, smoothing_slope=0.2, optimized=False)
fit1_2 = fit2.forecast(len(my_data)).rename("Holt's linear trend")

my_data.plot(kind="line",figsize=(10,5))
fit1_2.plot(kind="line",figsize=(10,5),color='orange')


# In[ ]:




