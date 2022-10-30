#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv")


# In[3]:


data.head()


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


data.describe()


# In[14]:


data['quality'].unique()   #output


# In[15]:


data['quality'].value_counts()  #multiclass 


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[18]:


X=data.drop("quality",axis=1)
y=data['quality']


# In[19]:


X.head()


# In[20]:


y.head()


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[26]:


scaler=StandardScaler()


# In[27]:


scaler.fit(X_train) #fiting the data, caculating means and sd


# In[29]:


print(scaler.mean_)


# In[32]:


X_train_tf=scaler.transform(X_train)


# In[33]:


X_train_tf


# In[34]:


y


# In[50]:


from sklearn.svm import SVC


# In[51]:


model=SVC()


# In[52]:


model.fit(X_train_tf,y_train)


# In[53]:


model.score(X_train_tf,y_train) #training accuracy


# In[56]:


X_test_tf=scaler.transform(X_test)


# In[59]:


y_predict=model.predict(X_test_tf)   #predicted values


# In[ ]:





# In[ ]:


y_test #actual values


# In[58]:


from sklearn.metrics import accuracy_score


# In[60]:


accuracy_score(y_test,y_predict)   #testing accuracy


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[65]:


model2=LogisticRegression()


# In[66]:


model2.fit(X_train_tf,y_train)


# In[69]:


y_predict2=model2.predict(X_test_tf)


# In[71]:


accuracy_score(y_test,y_predict2)


# In[75]:


model.predict([[-0.3536421 ,  0.15558944, -0.96737373, -0.03334372,  0.55556956,
       -0.18596079, -0.02314512,  0.1740298 , -0.48314224,  0.00685666,
       -0.76696884]])


# In[73]:


X_test_tf[0]


# In[76]:


#https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv


# In[ ]:




