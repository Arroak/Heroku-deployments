#!/usr/bin/env python
# coding: utf-8

# In[177]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[178]:


st.write('''
This app predicts the penguin species from the penguin data such as species, bill_length_mm, bill_depth_mm', 'flipper_length_mm,
       body_mass_g, sex_female, sex_male, island_Biscoe,
       island_Dream, island_Torgersen
''')


# In[179]:


st.sidebar.header('User input Features')
st.sidebar.markdown("""
Example csv input file penguins_example.csv
""")

#collecting user input
upload_file = st.sidebar.file_uploader('upload your input csv', type=['csv'])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)',32.1,59.6,43.9,)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)',13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)',172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)',2700.0,6300.0,4207.0)
        
        data = {'island':island,'sex':sex, 'flipper_length_mm':flipper_length_mm,
               'bill_length_mm':bill_length_mm,'bill_depth_mm':bill_depth_mm, 'body_mass_g':body_mass_g }
        
        features = pd.DataFrame(data,index=[0])
        return features
    input_df = user_input_features()
    


# In[180]:


#combining user's input with entire penguin dataset
penguins_raw = pd.read_csv('madagacar',index_col=0)
penguins = penguins_raw.drop(columns=['species'])


# In[181]:


penguins_raw.head()


# In[182]:


input_df.head()


# In[183]:


penguins_raw.head()


# In[184]:


penguins.head()


# In[185]:


df= pd.concat([input_df,penguins], axis=0)


# In[186]:


#encoding or ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
    
df = df[:1] # selects only the first row(i.e the users input data)


# In[187]:


# Display the user input features
st.subheader('User input Features')

if upload_file is not None:
    st.write(df)
    
else:
    st.write('Awaiting csv file to be uploaded, currently using example input parameters(shown below).')
    st.write(df)
    
    


# In[188]:


df.head()


# In[189]:


#reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

#apply model to make preditions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prdition Probability')
st.write(prediction_proba)


# In[ ]:




