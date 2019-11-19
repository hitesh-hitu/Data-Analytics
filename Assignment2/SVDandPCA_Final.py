#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
from scipy.linalg import svd
import pandas as pd
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Hitesh Kumar/Videos/iris.csv")

df = pd.read_csv("C:/Users/Hitesh Kumar/Videos/iris.csv", usecols = ['sepal_length','sepal_width','petal_length','petal_width'])
#type(df)
n = np.array(df)



#PCA on the dataset
pca = decomposition.PCA(n_components=2)
pca.fit(n)
y = pca.transform(n)




#SVD on the dataset

U ,s ,V = svd(n)
#print(U[0][1])
#print(s)
#print(V)
#print(len(V))


d1=y[0:50]
d2=y[51:101]
d3=y[101:]

#print(len(d1))

graphdata = (d1, d2, d3)
colors = ("red", "green", "purple")
groups = ("seriosa", "versicolor", "virginia")





# Create plot for PCA
fig = plt.figure()
ax = fig.add_subplot(1,1,1)


for data, color, group in zip(graphdata, colors, groups):
    #print("{}  {}".format(len(data),color))
    for i in range (len(data)):
        x, y = data[i][0],data[i][1]
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30,label=group)
    

plt.title('PCA on Iris Dataset')
plt.legend(loc=4,shadow =True)# to place the label ( 'best' can be used)
plt.show()


#Plot for SVD

k1=U[0:50]
k2=U[51:101]
k3=U[101:]

graphsvd = (k1,k2,k3)

svdfig = plt.figure()

svdfig.set_figheight(9)
svdfig.set_figwidth(13)
bx = svdfig.add_subplot(2,2,2)

for data, color, group in zip(graphsvd, colors, groups):
    #print("{}  {}".format(len(data),color))
    for i in range (len(data)):
        x, y = data[i][0],data[i][1]
        #print(x,y)
        bx.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
    bx.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30,label=group)


plt.title('SVD on Iris Dataset')
plt.legend(loc='best',shadow =True)# to place the label ( 'best' can be used)
plt.show()

    
















# In[ ]:





# In[ ]:




