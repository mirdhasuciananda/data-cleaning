#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[89]:


# import library dan modul
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[90]:


# load dataset
df = pd.read_csv('Iris_unclean.csv')
df


# In[91]:


df = df.drop('Unnamed: 0', axis=1)
df


# In[92]:


# cek ukuran dataset
df.shape


# In[93]:


# cek informasi dataset
df.info()


# In[94]:


# cek deskripsi statistik dataset
df.describe()


# In[95]:


# cek nilai yang hilang (missing value)
df.isnull().sum()


# ## <font color = 'green'> 1. Cek Kolom SepalLengthCm

# In[96]:


# cek deksrispi statistik
df['SepalLengthCm'].describe()


# In[97]:


# cek jumlah nilai NaN pada kolom SepalLengthCm
df['SepalLengthCm'].isnull().sum()


# In[101]:


# mencari nilai index dari missing value
index_nan = np.where(df['SepalLengthCm'].isna())
index_nan


# In[103]:


# menghapus baris yang memiliki missing value
df = df.dropna(axis=0)
df


# In[104]:


# cek ukuran dataset
df.shape


# ## <font color = 'green'> 2. Cek Kolom SepalWidthCm

# In[105]:


# cek deskripsi statistik 
df['SepalWidthCm'].describe()


# In[106]:


# mendeteksi outliers dengan boxplot pada kolom SepalWidthCm
from matplotlib import *
import sys
from pylab import *

plt.figure(figsize=(10,5))
sns.boxplot(df['SepalWidthCm'])
plt.annotate('Outlier', (df['SepalWidthCm'].describe()['max'],0.1), xytext = (df['SepalWidthCm'].describe()['max'],0.4),
             arrowprops = dict(facecolor = 'yellow'), fontsize = 13 )


# In[108]:


# membuat fungsi untuk melihat data outliers
def detect_outliers(df, x):
    Q1 = df[x].describe()['25%']
    Q3 = df[x].describe()['75%']
    IQR = Q3-Q1
    return df[(df[x] < Q1-1.5*IQR) | (df[x] > Q3+1.5*IQR)]


# In[109]:


# melihat data outliers dari kolom SepalWidthCm
detect_outliers(df, 'SepalWidthCm')


# In[110]:


# menghapus baris yang memiliki nilai outliers
df = df.drop(detect_outliers(df, 'SepalWidthCm').index, axis=0)
df


# In[111]:


# cek ukuran dataset
df.shape


# In[112]:


# cek ulang outlier dengan fungsi
detect_outlier(df, 'SepalWidthCm')


# In[113]:


# cek ulang outlier dengan boxplot
plt.figure(figsize = (10, 5))
sns.boxplot(df['SepalWidthCm'])


# In[ ]:




