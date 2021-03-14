#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV , train_test_split


# In[7]:


df1= pd.read_excel(r"C:\Users\prudhvi malladi\Desktop\air_test.xlsx")

df= pd.read_excel(r"C:\Users\prudhvi malladi\Desktop\air_train.xlsx")
df1


# In[8]:


df.dropna(inplace= True)
df1.dropna(inplace =True)


# In[9]:


df.drop(["Date_of_Journey" , "Dep_Time" , "Arrival_Time","Additional_Info","Route"] , axis =1 , inplace = True)
df1.drop(["Date_of_Journey" , "Dep_Time" , "Arrival_Time","Additional_Info","Route"] , axis =1 , inplace = True)


# In[10]:


le = LabelEncoder()
df["Source"] = le.fit_transform(df["Source"])
df["Destination"] = le.fit_transform(df["Destination"])

df1["Source"] = le.fit_transform(df1["Source"])
df1["Destination"] = le.fit_transform(df1["Destination"])


# In[12]:


print(df.Airline.value_counts())
print(df1.Airline.value_counts())


# In[13]:


#mapping

stop = {
    "non-stop":0,
    "1 stop":1,
    "2 stops":2,
    "3 stops":3,
    "4 stops":4
}

df.loc[: , "Total_Stops"] = df["Total_Stops"].map(stop)

df1.loc[: , "Total_Stops"] = df1["Total_Stops"].map(stop)


# In[14]:


df


# In[25]:


#change of duration into hr and min
#train

duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[26]:


duration = list(df1["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours_t = []
duration_mins_t = []
for i in range(len(duration)):
    duration_hours_t.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins_t.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[27]:


df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins


# In[28]:


df1["Duration_hours"] = duration_hours_t
df1["Duration_mins"] = duration_mins_t


# In[29]:


df1
df


# In[30]:



df.drop(["Duration"] , axis =1 , inplace =True)
df1.drop(["Duration"] , axis =1 , inplace =True)


# In[31]:


df.Airline.value_counts()


# In[32]:


df1.Airline.value_counts()


# In[33]:


df = df[df.Airline != 'Trujet']


# In[34]:


df


# In[35]:



df = df[df.Airline != 'Multiple carriers Premium economy']
df = df[df.Airline != 'Jet Airways Business']
df = df[df.Airline != 'Vistara Premium economy']



df1 = df1[df1.Airline != 'Multiple carriers Premium economy']
df1 = df1[df1.Airline != 'Jet Airways Business']
df1 = df1[df1.Airline != 'Vistara Premium economy']


# In[36]:


df


# In[37]:


df1.Airline.value_counts()


# In[38]:


df.Airline.value_counts()


# In[39]:


#mapping

stop = {
    "Jet Airways":0,
    "IndiGo":1,
    "Air India":2,
    "Multiple carriers":3,
    "SpiceJet":4 , "Vistara":5 ,"Air Asia":6 , "GoAir":7, 
}

df.loc[: , "Airline"] = df["Airline"].map(stop)
df1.loc[: , "Airline"] = df1["Airline"].map(stop)


# In[40]:


df.isna().sum()


# In[41]:


df1.isna().sum()


# In[ ]:





# In[42]:


x = df.drop(["Price"] , axis =1)
y = df.Price
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state = 100 , test_size = 0.3)


# In[43]:


df.dropna(inplace = True)
df.isna().sum()


# In[44]:


feat = ExtraTreesRegressor()
feat.fit(x_train , y_train)


# In[45]:


features = pd.Series( feat.feature_importances_ , index = x_train.columns )
features.nlargest(10).plot(kind = "barh")
plt.show()


# In[46]:


lr = LinearRegression()
rfr = RandomForestRegressor()
dt = DecisionTreeRegressor()


# In[47]:


print(lr.fit(x_train , y_train))
print(rfr.fit(x_train , y_train))
print(dt.fit(x_train , y_train))


# In[48]:


#train acc

print(r2_score(lr.predict(x_train) , y_train))
print(r2_score(rfr.predict(x_train) , y_train))
print(r2_score(dt.predict(x_train) , y_train))


# In[49]:


#train acc

print(r2_score(lr.predict(x_test) , y_test))
print(r2_score(rfr.predict(x_test) , y_test))
print(r2_score(dt.predict(x_test) , y_test))


# In[50]:


sb.distplot(rfr.predict(x_train) - y_train)


# In[ ]:




