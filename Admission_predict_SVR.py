#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv")


# In[2]:


data.head()


# In[5]:


data.columns


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


data.isnull().sum()


# In[24]:


data.rename(columns={'Chance of Admit ':'Chance of Admit'},inplace=True)
X=data.drop("Chance of Admit",axis=1)


# In[25]:


y=data['Chance of Admit']


# In[28]:


X.head(1)


# In[27]:


y.head(1)


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


scaler=StandardScaler()


# In[39]:


X_train_tf=scaler.fit_transform(X_train)


# In[37]:


X_test_tf=scaler.transform(X_test)


# In[35]:


from sklearn.svm import SVR


# In[36]:


model=SVR()


# In[40]:


model.fit(X_train_tf,y_train)


# In[43]:


model.score(X_train_tf,y_train)


# In[44]:


y_predict=model.predict(X_test_tf)   #predicted values


# In[49]:


#Performance Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,y_predict))


# In[50]:



print(mean_absolute_error(y_test,y_predict))


# In[52]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_predict)
score


# In[ ]:




