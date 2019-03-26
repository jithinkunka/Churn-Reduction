
# coding: utf-8

# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN   
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[ ]:


#Setting path
os.chdir("C:/Users/jithin/Desktop/Data Science/Project")


# In[ ]:


#loading train and test data
df_train = pd.read_csv("Train_data.csv")
df_test = pd.read_csv("Test_data.csv")


# In[ ]:


#missing value analysis
#storing number of missing values of each variable in dataframe
missing_train = pd.DataFrame(df_train.isnull().sum())
missing_test = pd.DataFrame(df_test.isnull().sum())


# In[ ]:


missing_train


# In[ ]:


#distribution of target variable
y = df_train["Churn"].value_counts()
sns.barplot(y.index, y.values)


# In[ ]:


#target variable w r t categorical variable
#state vs churn
df_train.groupby(["state", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10))


# In[ ]:


#areacode vs churn
df_train.groupby(["area code", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5))


# In[ ]:


#international plan vs churn
df_train.groupby(["international plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5))


# In[ ]:


#voice mail plan vs churn
df_train.groupby(["voice mail plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

# In[ ]:

#describing train data
df_train.describe()


# In[ ]:


#assigning levels to categorical varibales
for i in range(0, df_train.shape[1]):
    if(df_train.iloc[:,i].dtypes == 'object'):
        df_train.iloc[:,i] = pd.Categorical(df_train.iloc[:,i])
        df_train.iloc[:,i] = df_train.iloc[:,i].cat.codes
        
for i in range(0, df_test.shape[1]):
    if(df_test.iloc[:,i].dtypes == 'object'):
        df_test.iloc[:,i] = pd.Categorical(df_test.iloc[:,i])
        df_test.iloc[:,i] = df_test.iloc[:,i].cat.codes        


# In[ ]:


df_test.head(10)


# In[ ]:


#storing target variable
train_targets = df_train.Churn
test_targets = df_test.Churn


# In[ ]:


#combining train and test data for data prepocessing
combined = df_train.append(df_test)


# In[ ]:


print(combined.shape, df_train.shape, df_test.shape)


# In[ ]:


cnames = ["account length","area code","number vmail messages","total day minutes","total day calls","total day charge",
           "total eve minutes","total eve calls","total eve charge","total night minutes","total night calls",
           "total night charge","total intl minutes","total intl calls", "total intl charge", 
           "number customer service calls"]


# In[ ]:


df_corr = combined.loc[:,cnames]


# In[ ]:


#correlation analysis
#set height and width of plot
f , ax = plt.subplots(figsize = (7,5))

#generate correlation matrix
corr = df_corr.corr()

#plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap = sns.diverging_palette(220,10,as_cmap=True),
           square = True, ax=ax)


# In[ ]:


cat_names = ["state","phone number","international plan","voice mail plan","Churn"]


# In[ ]:


#chi square test of independence
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(combined['Churn'],combined[i]))
    print(p)


# In[ ]:


#dropping unnecessary variables
combined = combined.drop(["total day minutes", "total eve minutes", "total night minutes", "total intl minutes",
                    "phone number","Churn"], axis = 1)


# In[ ]:


combined.shape


# In[ ]:


cnames = ["account length","area code","number vmail messages","total day calls","total day charge",
           "total eve calls","total eve charge","total night calls","total night charge","total intl calls", 
          "total intl charge", "number customer service calls"]


# In[ ]:


#normalization
for i in cnames:
    print(i)
    combined[i] = (combined[i]-min(combined[i]))/(max(combined[i])-min(combined[i]))


# In[ ]:


combined.head(10)


# In[ ]:


combined.shape


# In[ ]:


#loading libraries for model
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[ ]:


#splitting combined data to train and test

train = combined[:3333]
test = combined[3333:]


# In[ ]:


#decision tree model
c50_model = tree.DecisionTreeClassifier(criterion = 'entropy').fit(train, train_targets)

c50_pred = c50_model.predict(test)


# In[ ]:


c50_pred


# In[ ]:


#dot file to look at decision tree
dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(c50_model, out_file=dotfile, feature_names = train.columns)


# In[ ]:


#testing accuracy of model
from sklearn.metrics import confusion_matrix

CM = pd.crosstab(test_targets, c50_pred)
CM


# In[ ]:


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

accuracy_score(test_targets,c50_pred)*100

#accuracy = 92.32153

#(FN*100)/(FN+TP)

#FNR = 32.142857

#(TP*100)/(TP+FN)

#Recall = 67.8571428


# In[ ]:


#random forest model

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 100).fit(train,train_targets)

RF_prediction = RF_model.predict(test)


# In[ ]:


RF_prediction


# In[ ]:


CM = pd.crosstab(test_targets, RF_prediction)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#accuracy = 94.96100

#(FN*100)/(FN+TP)
#FNR = 33.928571


# In[ ]:


#KNN implementation
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(train, train_targets)


# In[ ]:


#predict test cases
KNN_Predictions = KNN_model.predict(test)


# In[ ]:


#build confusion matrix
CM = pd.crosstab(test_targets, KNN_Predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#accuracy = 86.622675


#False Negative rate 
#(FN*100)/(FN+TP)
#FNR =97.321428


# In[ ]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(train, train_targets)


# In[ ]:


#predict test cases
NB_Predictions = NB_model.predict(test)


# In[ ]:


#Build confusion matrix
CM = pd.crosstab(test_targets, NB_Predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(Y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#Accuracy = 85.842831433

#False Negative rate 
#(FN*100)/(FN+TP)
#FNR = 60.2678571


# In[ ]:


#we will be fixing random forest model as it provides best results
#now we will generate example out for out sample input test data with Random forest predictions 

move = pd.DataFrame(RF_prediction)

move = move.rename(columns = {0:'move'})


# In[ ]:


test = test.join(move['move'])


# In[ ]:


test.to_csv("example_output.csv", index = False)
