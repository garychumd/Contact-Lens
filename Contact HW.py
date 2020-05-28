#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

#import all the required libraries.  Must run this or functions will not work


# In[2]:


df = pd.read_csv('/Users/Harrison/Desktop/ContactHW.csv')


# In[3]:


df.head()


# In[14]:


# create a data array only with data with TypeAge30 = 2?

Contact = df.loc[df['TypeAt30'] == 2]

plt.scatter(Contact.Age18Acuity, Contact.Age30Acuity,  color='red')
plt.xlabel("Age 18 Visual Acuity")
plt.ylabel("Age 30 Acuity")
plt.title("Wearing Contacts")
plt.show()


# In[15]:


# split data set into train and test sets. 80% for training. 20% for testing
# ~ invert all bits (not function)
msk = np.random.rand(len(df)) < 0.8
train = df  # assign train to random 80% of test data
test = df[~msk]  # assign all data not in msk

#import the model
from sklearn import linear_model

# Train regression model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Age18Acuity']])
train_y = np.asanyarray(train[['Age30Acuity']])
regr.fit (train_x, train_y)

# Print the coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[16]:


#print the fit line over the data

#print scatter graph of the data
plt.scatter(train.Age18Acuity, train.Age30Acuity,  color='red') 

#print the fit line
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

#data label
plt.xlabel("Age 18 Vision Acuity")
plt.ylabel("Age 30 Vision")


# In[ ]:




