#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[1]:


# import library dan modul
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from matplotlib import *
import sys
from pylab import *


# In[2]:


# load dataset
df = pd.read_csv('Iris_unclean.csv')
df


# In[3]:


# menghapus kolom 'Unnamed'
df = df.drop('Unnamed: 0', axis=1)
df


# In[4]:


# cek ukuran dataset
df.shape


# In[5]:


# cek informasi dataset
df.info()


# In[6]:


# cek deskripsi statistik dataset
df.describe()


# In[7]:


# cek nilai yang hilang (missing value)
df.isnull().sum()


# ## <font color = 'green'> 1. Cek Kolom SepalLengthCm

# In[8]:


# cek deksrispi statistik
df['SepalLengthCm'].describe()


# In[9]:


# cek jumlah nilai NaN pada kolom SepalLengthCm
df['SepalLengthCm'].isnull().sum()


# In[10]:


# mencari nilai index dari missing value
index_nan = np.where(df['SepalLengthCm'].isna())
index_nan


# In[11]:


# menghapus baris yang memiliki missing value
df = df.dropna(axis=0)
df


# In[12]:


# cek ukuran dataset
df.shape


# In[13]:


# membuat fungsi untuk melihat data outliers
def detect_outliers(df, x):
    Q1 = df[x].describe()['25%']
    Q3 = df[x].describe()['75%']
    IQR = Q3-Q1
    return df[(df[x] < Q1-1.5*IQR) | (df[x] > Q3+1.5*IQR)]


# In[14]:


# cek outlier dari kolom SepalLengthCm
detect_outliers(df, 'SepalLengthCm')


# ## <font color = 'green'> 2. Cek Kolom SepalWidthCm

# In[15]:


# cek deskripsi statistik 
df['SepalWidthCm'].describe()


# In[16]:


# melihat data outliers dari kolom SepalWidthCm
detect_outliers(df, 'SepalWidthCm')


# In[17]:


# mendeteksi outliers dengan boxplot pada kolom SepalWidthCm
plt.figure(figsize=(10,5))
sns.boxplot(df['SepalWidthCm'])
plt.annotate('Outlier', (df['SepalWidthCm'].describe()['max'],0.1), xytext = (df['SepalWidthCm'].describe()['max'],0.4),
             arrowprops = dict(facecolor = 'yellow'), fontsize = 13 )


# In[18]:


# menghapus baris yang memiliki nilai outliers
df = df.drop(detect_outliers(df, 'SepalWidthCm').index, axis=0)
df


# In[19]:


# cek ukuran dataset
df.shape


# In[31]:


# cek ulang outlier dengan fungsi
detect_outliers(df, 'SepalWidthCm')


# In[21]:


# cek ulang outlier dengan boxplot
plt.figure(figsize = (10, 5))
sns.boxplot(df['SepalWidthCm'])


# ## <font color = 'green'> 3. Cek Kolom PetalLengthCm

# In[22]:


# Cek deskripsi statistik PetalLengthCm
df['PetalLengthCm'].describe()


# In[23]:


# periksa nilai negatif PetalLengthCm
df[df['PetalLengthCm']<0]


# In[24]:


# menghapus baris yang memiliki nilai negatif
df = df.drop(df[df['PetalLengthCm']<0].index, axis=0)
df


# In[25]:


# cek ukuran dataset
df.shape


# In[26]:


# cek outlier dari kolom PetalLengthCm
detect_outliers(df, 'PetalLengthCm')


# ## <font color = 'olive'> Cek Dataset Setelah Proses Cleaning

# In[27]:


# cek ulang informasi dataset
df.info()


# In[28]:


# cek ulang deskripsi statistik datset
df.describe()


# In[29]:


# cek ulang nilai missing value
df.isnull().sum()


# In[30]:


# menampikan 10 baris awal dataset setelah proses cleaning
df.head(10)

