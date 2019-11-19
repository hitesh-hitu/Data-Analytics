#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import statistics as stat
import pandas as pd
import seaborn as sns

df = pd.read_csv('2019_Results.csv',encoding='ISO-8859-1')
df.head()  #displays first 5 rows

res = pd.read_csv('2019_Results_Winning_Candidate.csv',encoding='ISO-8859-1')
#df
#df.tail()    #displays last 5 rows
#df.describe()   #Provides summary measures for variables of type : int or float
#df.dtypes  #displays type of each variable
#df.columns  #displays variable names aka. Column name
#df.columns[0]  #displays 1st column name


# In[2]:


print('Mean ', np.mean(df))


# In[3]:


party_list = df['Party'].value_counts()
#print(party_list['Dravida Munnetra Kazhagam']) 
#print(type(party_list))
#print(party_list.index)
majorparties = []
majorpartiesfreq = []

partywinlist = res['Party'].value_counts()
#print(partywinlist)

majorparties.append('Bharatiya Janata \nParty')
majorparties.append('Indian National \nCongress')
majorparties.append('Dravida Munnetra \nKazhagam')
majorparties.append('All India Trinamool \nCongress')
majorparties.append('Yuvajana Sramika \nParty')
majorparties.append('Shivsena')

majorpartiesfreq.append(party_list['Bharatiya Janata Party'])
majorpartiesfreq.append(party_list['Indian National Congress'])
majorpartiesfreq.append(party_list['Dravida Munnetra Kazhagam'])
majorpartiesfreq.append(party_list['All India Trinamool Congress'])
majorpartiesfreq.append(party_list['Yuvajana Sramika Rythu Congress Party'])
majorpartiesfreq.append(party_list['Shivsena'])


#print(majorparties)
#print(majorpartiesfreq)

majorpartywinlist = []

majorpartywinlist.append(partywinlist['Bharatiya Janata Party'])
majorpartywinlist.append(partywinlist['Indian National Congress'])
majorpartywinlist.append(partywinlist['Dravida Munnetra Kazhagam'])
majorpartywinlist.append(partywinlist['All India Trinamool Congress'])
majorpartywinlist.append(partywinlist['Yuvajana Sramika Rythu Congress Party'])
majorpartywinlist.append(partywinlist['Shivsena'])

legend = ["Contested","Won"]
index = np.arange(len(majorparties))
plt.bar(index, majorpartiesfreq , width=0.5)
plt.bar(index, majorpartywinlist ,width=0.5)
plt.xlabel('Party', fontsize=7)
plt.ylabel('Number of Contestants', fontsize=7)
plt.xticks(index, majorparties, fontsize=8, rotation=10)
plt.legend(legend,loc=1)
plt.show()


# In[ ]:


flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights)


# In[ ]:




