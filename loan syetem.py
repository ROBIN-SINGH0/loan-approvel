#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv("loan_approval_data.csv")


# In[5]:


data.sample(10)


# In[7]:


data.isnull().sum()


# In[11]:


data.info()


# In[13]:


data.describe()


# # **MISING VALUES**

# In[16]:


categorical_colms=data.select_dtypes(include=["object"]).columns
numrical_colms=data.select_dtypes(include=["number"]).columns


# In[18]:


categorical_colms,numrical_colms


# In[20]:


from sklearn.impute import SimpleImputer
num_imp=SimpleImputer(strategy="mean")
data[numrical_colms]=num_imp.fit_transform(data[numrical_colms])


# In[21]:


from sklearn.impute import SimpleImputer
cate_imp=SimpleImputer(strategy="most_frequent")
data[categorical_colms]=cate_imp.fit_transform(data[categorical_colms])


# In[24]:


data.isnull().sum()


# # **EDA**

# In[27]:


#Analysise tha data how much loan is approved
approved_cnt=data["Loan_Approved"].value_counts()


# In[29]:


approved_cnt


# In[31]:


plt.pie(approved_cnt, labels=["NO","YES"],autopct="%1.1f%%")
plt.title("loan is approved or not")


# **Let's analysis every category** 

# In[34]:


ax=sns.barplot(data["Gender"].value_counts())
ax.bar_label(ax.containers[0])


# In[36]:


ax=sns.barplot(data["Marital_Status"].value_counts())
ax.bar_label(ax.containers[0])


# In[38]:


ax=sns.barplot(data["Employment_Status"].value_counts())
ax.bar_label(ax.containers[0])


# In[40]:


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

sns.barplot(
    x=data["Gender"].value_counts().index,
    y=data["Gender"].value_counts().values,
    ax=axes[0, 0]
)
axes[0, 0].bar_label(axes[0, 0].containers[0])

sns.barplot(
    x=data["Marital_Status"].value_counts().index,
    y=data["Marital_Status"].value_counts().values,
    ax=axes[0, 1]
)
axes[0, 1].bar_label(axes[0, 1].containers[0])

sns.barplot(
    x=data["Employment_Status"].value_counts().index,
    y=data["Employment_Status"].value_counts().values,
    ax=axes[1, 0]
)
axes[1, 0].bar_label(axes[1, 0].containers[0])

sns.barplot(
    x=data["Property_Area"].value_counts().index,
    y=data["Property_Area"].value_counts().values,
    ax=axes[1, 1]
)
axes[1, 1].bar_label(axes[1, 1].containers[0])

plt.tight_layout()
plt.show()


# In[42]:


ax=sns.histplot(data["Applicant_Income"],bins=10)


# In[44]:


fig,axes=plt.subplots(2,2)
sns.boxplot(ax=axes[0,0],data=data,x="Loan_Approved",y="Applicant_Income")
sns.boxplot(ax=axes[0,1],data=data,x="Loan_Approved",y="Credit_Score")
sns.boxplot(ax=axes[1,0],data=data,x="Loan_Approved",y="DTI_Ratio")
sns.boxplot(ax=axes[1,1],data=data,x="Loan_Approved",y="Savings")
plt.tight_layout()


# In[46]:


sns.histplot(x="Credit_Score",hue="Loan_Approved",data=data,multiple='dodge')


# In[48]:


data=data.drop("Applicant_ID",axis=1)


# In[50]:


data


# # ENCODING

# In[53]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder


# In[55]:


data.info()


# # LABEL ENCODING

# In[58]:


lab_en=LabelEncoder()
data["Education_Level"]=lab_en.fit_transform(data["Education_Level"])
data["Loan_Approved"]=lab_en.fit_transform(data["Loan_Approved"])


# In[60]:


one_hot_col=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender", "Employer_Category"]


# # ONEHOTENCODING

# In[63]:


one_hot_enc=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")
encoded=one_hot_enc.fit_transform(data[one_hot_col])


# In[65]:


encoded_data=pd.DataFrame(encoded,columns=one_hot_enc.get_feature_names_out(one_hot_col),index=data.index)


# In[67]:


new_data=pd.concat([data.drop(columns=one_hot_col),encoded_data],axis=1)


# In[ ]:





# In[70]:


new_data.head()


# # **HEATMAP**

# In[73]:


data.head()


# In[75]:


num_col=new_data.select_dtypes(include="number")
corr_mat=num_col.corr()


# In[77]:


num_col.corr()["Loan_Approved"].sort_values(ascending=False)


# In[79]:


plt.figure(figsize=(15,8))
sns.heatmap(corr_mat,annot=True,fmt=".2f",cmap="coolwarm")


# # FEATURE SCALING & TRAIN TEST SPLIT

# In[82]:


X=new_data.drop(columns="Loan_Approved")
y=new_data["Loan_Approved"]


# In[84]:


X.head()


# In[86]:


y.head()


# In[88]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# # SCALING

# In[91]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# # LOGISTIC REGRESSION

# In[94]:


#FOR LOGISTICREGRESION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
log_model= LogisticRegression()
log_model.fit(X_train_scaled,y_train)
y_pred=log_model.predict(X_test_scaled)

print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# # KNN

# In[97]:


#FOR KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
KNN_model= KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_scaled,y_train)
y_pred=KNN_model.predict(X_test_scaled)


print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# # NAIVE BAYE'S

# In[100]:


#FOR NAIVE 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
NAIVE_model= GaussianNB()
NAIVE_model.fit(X_train_scaled,y_train)
y_pred=NAIVE_model.predict(X_test_scaled)


print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# # **SO FOR THIS PROBLEM OOUR NAIVE BAYE'S IS BEST ALGO**

# # FEATURE ENGINEERING
# 

# In[117]:


new_data.info()


# In[119]:


new_data["DTI_Ratio_sq"]=new_data["DTI_Ratio"]**2
new_data["Credit_Score_sq"]=new_data["Credit_Score"]**2

X=new_data.drop(columns=["Loan_Approved","DTI_Ratio","Credit_Score"])
y=new_data["Loan_Approved"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#for LogisticRegression
log_model= LogisticRegression()
log_model.fit(X_train_scaled,y_train)
y_pred=log_model.predict(X_test_scaled)


print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# In[121]:


#FOR KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
KNN_model= KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_scaled,y_train)
y_pred=KNN_model.predict(X_test_scaled)


print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# In[123]:


#FOR NAIVE 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
NAIVE_model= GaussianNB()
NAIVE_model.fit(X_train_scaled,y_train)
y_pred=NAIVE_model.predict(X_test_scaled)


print("precision score is :", precision_score(y_test,y_pred))
print("recall_score is :", recall_score(y_test,y_pred))
print("f1_score is :", f1_score(y_test,y_pred))
print("accuracy score is :", accuracy_score(y_test,y_pred))
print("confusion_matrix is :", confusion_matrix(y_test,y_pred))


# In[1]:


get_ipython().system('pip install streamlit')


# In[3]:


get_ipython().system('streamlit --version')


# In[ ]:




