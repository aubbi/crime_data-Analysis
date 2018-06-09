
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[2]:

df=pd.read_csv(r'C:\Users\HP\Desktop\Crimes_-_2001_to_present.csv')


# In[4]:

df.dropna(axis = 0, how = 'any', inplace = True)


# In[ ]:




# In[6]:

print(df.dtypes)


# In[7]:

df.Date = pd.to_datetime(df.Date)


# In[8]:

print(df.dtypes)


# In[12]:

def doKMeans(dataframe):
  df = pd.concat([dataframe.Longitude, dataframe.Latitude], axis = 1)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x = df.Longitude, y = df.Latitude, marker='.', alpha=0.3, s = 30)
  kmeans_model = KMeans(n_clusters = 6, init = 'random', n_init = 60, max_iter = 360, random_state = 43)
  labels = kmeans_model.fit_predict(df)
  centroids = kmeans_model.cluster_centers_
  ax.scatter(x = centroids[:,0], y = centroids[:,1], marker='x', c='blue', alpha=0.5, linewidths=3, s = 129)
  print (centroids)
  plt.show()
doKMeans(df)


# In[13]:

#crimes after 2011
df2 = df[df.Date > '2011-01-01']
doKMeans(df2)
plt.title("Dates limited to 2011 and later")
plt.show()


# In[ ]:



