#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

import os


# In[14]:


iris=pd.read_csv("C:/Users/mymai/Downloads/CAR.csv")


# In[15]:


iris.head()


# In[16]:


iris['selling_price']<300000


# In[17]:


iris[iris['selling_price']<300000]


# In[18]:


y=[['selling_price']]


# In[19]:


x=[['km_driven']]


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


le= LabelEncoder()


# In[24]:


iris.fuel=le.fit_transform(iris.fuel)


# In[25]:


iris.transmission=le.fit_transform(iris.transmission)


# In[26]:


iris.seller_type=le.fit_transform(iris.seller_type)


# In[27]:


iris.owner=le.fit_transform(iris.owner)


# In[28]:


iris.name=le.fit_transform(iris.name)


# In[29]:


iris.head()


# In[30]:


iris.fuel=iris.fuel.astype('category')


# In[31]:


iris.seller_type=iris.seller_type.astype('category')


# In[32]:


iris.transmission=iris.transmission.astype('category')


# In[33]:


iris.owner=iris.owner.astype('category')


# In[34]:


iris.name=iris.name.astype('category')


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[36]:


y=[['selling_price']]


# In[37]:


x=[['km_driven']]


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[40]:


iris.describe()


# In[41]:


iris.describe('all')


# In[42]:


iris.describe(include='all')


# In[43]:


x_train.head()


# In[44]:


y=iris[['selling_price']]


# In[45]:


x=iris[['name','year','km_driven','fuel','seller_type','transmission','owner']]


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


lr2=LinearRegression()


# In[50]:


lr2.fit(x_train,y_train)


# In[51]:


y_pred=lr2.predict(x_test)


# In[52]:


y_pred[0:5]


# In[53]:


from sklearn.metrics import mean_squared_error


# In[54]:


mean_squared_error(y_test,y_pred)


# In[ ]:




