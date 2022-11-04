#!/usr/bin/env python
# coding: utf-8

# 
# ___
# # Logistic Regression Project - Solutions
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


import pickle


# In[2]:


ad_data = pd.read_csv('advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 

# ** Split the data into training set and testing set using train_test_split**

# In[9]:


from sklearn.model_selection import train_test_split


# In[16]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ** Train and fit a logistic regression model on the training set.**

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[20]:


predictions = logmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[21]:


from sklearn.metrics import classification_report


# In[22]:


print(classification_report(y_test,predictions))


# In[25]:


pickle.dump(logmodel, open('logr.pkl', 'wb'))


# In[ ]:




