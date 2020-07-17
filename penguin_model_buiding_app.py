#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score


# In[36]:


penguins = pd.read_csv('madagacar', index_col=0)
penguins.head()


# In[37]:


df = penguins.copy()


# In[38]:


target = 'species'
encode = ['sex','island']


# In[39]:


for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]


# In[40]:


target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df[target]= df[target].apply(target_encode)


# In[41]:


# seaparating X and Y
X = df.drop(target, axis=1)
Y =  df[target]


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=21)


# In[43]:


#model
clf = RandomForestClassifier(n_estimators=1000,max_features='sqrt')
clf.fit(x_train,y_train)
predict = clf.predict(x_test)


# In[44]:


print('confusion matrix: ', confusion_matrix(y_test,predict))
print('accuracy score: ', accuracy_score(y_test,predict))


# In[45]:


#saving the model
pickle.dump(clf,open('penguins_clf.pkl','wb'))


# In[ ]:




