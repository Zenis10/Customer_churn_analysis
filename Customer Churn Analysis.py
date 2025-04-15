#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER CHURN ANALYSIS

# In[ ]:


pip install pandas numpy matplotlib seaborn scikit-learn 


# # LOADING LIBRARIES

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier  


# READING DATASET

# In[3]:


data = pd.read_csv(r'C:\Users\zenis\Desktop\project\Telco-Customer-Churn.csv')  
print(data.head())  


# CHECKING FOR MISSING VALUES

# In[4]:


print("Missing values in each column:")  
print(data.isnull().sum())  


# SUMMARY STATISTICS

# In[5]:


print("\nSummary statistics:")  
print(data.describe()) 


# VISUALIZING CHURN DISTRIBUTION

# In[6]:


sns.countplot(x='Churn', data=data)  
plt.title('Churn Distribution')  
plt.show() 


# In[7]:


data.dropna(inplace=True)  


# ENCODING CATEGORICAL VARIABLES

# In[10]:


label_encoders = {}  
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:  
    le = LabelEncoder()  
    data[col] = le.fit_transform(data[col])  
    label_encoders[col] = le


# FEATURE SELECTION

# In[12]:


X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn'] 


# SPLIT DATA

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


# FEATURE SCALING

# In[15]:


scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   


# MODEL TRAINING

# In[17]:


model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train)


# MODEL PREDICTION

# In[19]:


# Predict on the test set  
y_pred = model.predict(X_test)    


# MODEL EVALUATION

# In[20]:


print(confusion_matrix(y_test, y_pred))  # Confusion Matrix  
print(classification_report(y_test, y_pred))  # Precision, Recall, F1-Score  


# VISUALIZATION OF RESULTS

# In[22]:


cm = confusion_matrix(y_test, y_pred)  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
plt.title('Confusion Matrix')  
plt.xlabel('Predicted')  
plt.ylabel('Actual')  
plt.show() 


# In[ ]:




