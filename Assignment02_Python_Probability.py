#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[10]:


file_path='https://github.com/Accelerate-AI/Data-Science-Global-Bootcamp/raw/main/01%20Python/gapminder.csv'
import pandas as pd
df = pd.read_csv(file_path, error_bad_lines=False)
df


# In[12]:


df.info


# In[15]:


df.describe()


# In[17]:


df.dtypes


# In[20]:


# One hot encoding using pandas
one_hot_encoded_data = pd.get_dummies(df, columns = ['Region'])

one_hot_encoded_data


# In[21]:


#Label encoding using pandas

# Converting type of columns to 'category'
df['Region_lab'] = df['Region'].astype('category')

# Assigning numerical values and storing in another column
df['Region_lab'] = df['Region_lab'].cat.codes

df


# In[22]:


# Based on the above two encoding method results, i will use One hot encoding technique. 
# As the column 'Region' is not ordinal and the number of categories are not very high.


# In[23]:


file_path='https://github.com/Accelerate-AI/Data-Science-Global-Bootcamp/raw/main/01%20Python/wine_data_UCI.csv'
import pandas as pd
df1 = pd.read_csv(file_path, error_bad_lines=False)
df1


# In[24]:


df1.info()


# In[25]:


df1.describe()


# In[31]:


# Yes, fearure scaling is required. As we see the features have diffrent min max range but the units are not mentioned in the data set. 
# Hence its important to bring the column values to a standard/normalized scale.


# In[34]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


# In[35]:


df1_norm = (df1 - df1.min())/ (df1.max() - df1.min())
df1_norm


# In[36]:


fig, ax=plt.subplots(1,2)
sns.kdeplot(df1.quality, ax=ax[0], color='y',shade=True, bw=0.5,)
ax[0].set_title("Original Data")
sns.kdeplot(df1_norm.quality, ax=ax[1], shade=True, bw=0.5, color="blue")
ax[1].set_title("Normalized data")
plt.show()


# In[37]:


fig, ax=plt.subplots(1,2)
sns.kdeplot(df1.Alcohol, ax=ax[0], color='y',shade=True, bw=0.5,)
ax[0].set_title("Original Data")
sns.kdeplot(df1_norm.Alcohol, ax=ax[1], shade=True, bw=0.5, color="blue")
ax[1].set_title("Normalized data")
plt.show()


# In[38]:


fig, ax=plt.subplots(1,2)
sns.kdeplot(df1.Malicacid, ax=ax[0], color='y',shade=True, bw=0.5,)
ax[0].set_title("Original Data")
sns.kdeplot(df1_norm.Malicacid, ax=ax[1], shade=True, bw=0.5, color="blue")
ax[1].set_title("Standardized data")
plt.show()


# In[41]:


# Z score normalization
# Calculating the mean and standard deviation
df1_Znorm = (df1 - df1.mean())/df1.std()
print(df1_Znorm)


# In[43]:


fig, ax=plt.subplots(1,2)
sns.kdeplot(df1.quality, ax=ax[0], color='y',shade=True, bw=0.5,)
ax[0].set_title("Original Data")
sns.kdeplot(df1_Znorm.quality, ax=ax[1], shade=True, bw=0.5, color="blue")
ax[1].set_title("Standardized data")
plt.show()


# In[44]:


fig, ax=plt.subplots(1,2)
sns.kdeplot(df1.quality, ax=ax[0], color='y',shade=True, bw=0.5,)
ax[0].set_title("Original Data")
sns.kdeplot(df1_Znorm.quality, ax=ax[1], shade=True, bw=0.5, color="blue")
ax[1].set_title("Standardized data")
plt.show()


# In[45]:


from matplotlib import pyplot
# histograms of the variables
df1.hist()
pyplot.show()


# In[46]:


# 3 Answer:
# From the above plot we can see the data set have differnt min and Max values and not all the columns follow normal/gussian distribution.
# As a result, I would go with Min_Max scale normalization for this data set.


# In[47]:


# 4 Answer
# A = event rater is diligent 
# B =event rater is non-dilegent 
# C = event that rater has labeled a piece as non spam
# p(A)= 0.9 
# p(B)= 0.1 
# p(C|A) = 0.95 
# p(C|B) =0.5 
# P(been rated as non-spam, what is the probability that is it actually non-spam?)=p(Rater is Deligent| Rater has identified the piece as Non Spam) 
# = P(c|A) P(A)/(p(C|A)P(A) + p(c|B)P(B)) => 0.950.9/(0.950.9 + 0.1 0.5) = 0.94

# Hence the answer is 94% of the time.


# In[50]:


# 5 Answer:
# As given we are saying the probability of seeing any other cars in 30 minutes is 95% or more clearly,
#that implies the probability of not seeing any other cars is 5%.

# Lets assume that the cars are randomly distributed.

# In order that we do not see a car for 30 minutes, it is necessary that we don't see another car for 10 minutes 3 times in a row. 
# Just as the probability of tossing a coin and getting 3 heads in a row is given by ½^3

# P(not30)=P(not10)^3

# Where P(not30) and P(not10) are the probabilities of not seeing a car in 30 and 10 minutes respectively. 
# We know P(not30) is 5% or 0.05 and the equation is now solvable by taking the cube root of 0.05

# P(not10) = cube root (0.05) = 3√0.05 = 0.3684 appx

# The ask is what is the probability of seeing a car in 10 minutes, we have worked out the probability of not seeing a car
# so we need 1 - 0.3684

# The chance of observing a car in a ten-minute period is .6316 (Answer).


# In[51]:


#6 
# Let X be the number of defectives when n items are packed into a box.
# P(X = 0) = (0.99)n
# P(X ≥ 1) = 1 – P(X = 0) = 1 − (0.99)n
# P(X ≥ 1) < 0.5
# ⇒ 1 − (0.99)n < 0.5
# ⇒ n < log 0.5/ log 0.99 = 68.97
# ⇒ n = 68


# In[ ]:





# In[ ]:




