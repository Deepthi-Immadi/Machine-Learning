#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Import DataSet

# In[2]:


df = pd.read_csv('houseprices.csv')
df


# # EDA

# In[3]:


df.isnull().sum()


# In[4]:


df.isna().any()


# # Plot the graph

# In[17]:


plt.xlabel('area(sqr ft)')
plt.ylabel('price(US $)')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[18]:


new_df=df.drop('price',axis='columns')
new_df


# In[23]:


new_df.shape


# In[25]:


price=df.price
price


# In[26]:


from sklearn import linear_model


# In[27]:


import seaborn as sns
sns.pairplot(df)
plt.show() 


# In[28]:


model=linear_model.LinearRegression()
model.fit(new_df,price)


# In[31]:


model.predict([[3300]])


# In[32]:


model.coef_


# In[33]:


model.intercept_


# # y=m *  X + b (m is the coefficient,b is the intercept)

# In[35]:


3300*135.78767123+180616.43835616432


# In[ ]:




