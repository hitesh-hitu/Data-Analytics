#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statistics as st


# In[6]:


res = pd.read_csv('2019_Results_Winning_Candidate.csv',encoding='ISO-8859-1')
res.head()
print(type(res))


# In[7]:


res['Party'].unique()


# In[17]:


s = res['State']
#print(s)
q = res['Party']
#print(q)
#q = s['Karnataka']
#print(q)

#x = q.combine_first(s)
#rint(x)
number= []
for i in range(0,541):
    if(s[i] == "Karnataka"):
         number.append(q[i])
unique=[]
unique=set(number)
#print(number)
#print(kar)

from collections import Counter

kar = dict(Counter(number))


number1= []
for i in range(0,541):
    if(s[i] == "Uttar Pradesh"):
         number1.append(q[i])

up = dict(Counter(number1))


number2= []
for i in range(0,541):
    if(s[i] == "Maharashtra"):
         number2.append(q[i])

mh = dict(Counter(number2))



number3= []
for i in range(0,541):
    if(s[i] == "West Bengal"):
         number3.append(q[i])

WB = dict(Counter(number3))

number4= []
for i in range(0,541):
    if(s[i] == "Bihar"):
         number4.append(q[i])

bi = dict(Counter(number4))


#print(number.count('Bharatiya Janata Party'))
#for i in len(number):
 #   if(Independent==number[i])

karn=list(kar.values())
up1=list(up.values()) 
mh1=list(mh.values())
wb1=list(WB.values())
bi1=list(bi.values())
#print(type(karn[0]))

majorparties = []

majorparties.append('Bharatiya Janata \nParty')
majorparties.append('Indian National \nCongress')
majorparties.append('Janata Dal (Secular)')
majorparties.append('Independent')
print(len(karn))
print(len(up1))
print(len(mh1))
index = np.arange(4)
plt.bar(index, karn, width=0.5)
plt.bar(index,up1,width=0.5 )
plt.bar(index,mh1, width=0.5)
plt.xlabel('Party', fontsize=7)
plt.ylabel('Number of Seats Won', fontsize=7)
plt.xticks(index,majorparties , fontsize=8, rotation=10)
plt.show()

print(kar.keys())
"""
print(karn)
print(up1)
print(mh1)
print(wb1)
print(bi1)
"""


# In[ ]:





# In[ ]:




