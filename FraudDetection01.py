#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Importing the libraries
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[51]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')


# In[52]:


credit_card_data.head()


# In[53]:


credit_card_data.tail()


# In[54]:


# dataset information
credit_card_data.info()


# In[55]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[56]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[57]:


# Checking the % distribution of normal vs fraud
classes=credit_card_data['Class'].value_counts()
normal_share=classes[0]/credit_card_data['Class'].count()*100
fraud_share=classes[1]/credit_card_data['Class'].count()*100

print(normal_share)
print(fraud_share)


# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[58]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[59]:


print(legit.shape)
print(fraud.shape)


# In[60]:


# statistical measures of the data
legit.Amount.describe()


# In[61]:


fraud.Amount.describe()


# In[62]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Under-Sampling

# In[63]:


legit_sample = legit.sample(n=393)


# In[64]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[65]:


new_dataset.head()


# In[66]:


new_dataset.tail()


# In[67]:


new_dataset['Class'].value_counts()


# In[68]:


new_dataset.groupby('Class').mean()


# Split the data into Training data & Testing Data

# In[132]:


# Splitting the dataset into X and y
Y= credit_card_data['Class']
X = credit_card_data.drop(['Class'], axis=1)


# In[134]:


X.head()


# In[135]:


Y.head()


# In[136]:


# Splitting the dataset using train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=100, test_size=0.20)


# Model Building

# In[123]:


#Lets perfrom StratifiedKFold and check the results
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_SKF_cv, X_test_SKF_cv = X.iloc[train_index], X.iloc[test_index]
    Y_train_SKF_cv, Y_test_SKF_cv = Y.iloc[train_index], Y.iloc[test_index]


# In[124]:


credit_card_data_Results = pd.DataFrame(columns=['Methodology', 'Model', 'Accuracy', 'roc_value', 'threshold'])


# In[125]:


# Created a common function to plot confusion matrix
def Plot_confusion_matrix(Y_test, pred_test):
  cm = confusion_matrix(Y_test, pred_test)
  plt.clf()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
  categoryNames = ['Non-Fraudalent','Fraudalent']
  plt.title('Confusion Matrix - Test Data')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  ticks = np.arange(len(categoryNames))
  plt.xticks(ticks, categoryNames, rotation=45)
  plt.yticks(ticks, categoryNames)
  s = [['TN','FP'], ['FN', 'TP']]
  
  for i in range(2):
      for j in range(2):
          plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),fontsize=12)
  plt.show()


# In[126]:


# # Created a common function to fit and predict on a Logistic Regression model for both L1 and L2
def buildAndRunLogisticModels(credit_card_data_Results, Methodology, X_train,Y_train, X_test, Y_test ):

  # Logistic Regression
  from sklearn import linear_model
  from sklearn.model_selection import KFold

  num_C = list(np.power(10.0, np.arange(-10, 10)))
  cv_num = KFold(n_splits=10, shuffle=True, random_state=42)

  searchCV_l2 = linear_model.LogisticRegressionCV(
          Cs= num_C
          ,penalty='l2'
          ,scoring='roc_auc'
          ,cv=cv_num
          ,random_state=42
          ,max_iter=10000
          ,fit_intercept=True
          ,solver='newton-cg'
          ,tol=10
      )

  searchCV_l1 = linear_model.LogisticRegressionCV(
          Cs=num_C
          ,penalty='l1'
          ,scoring='roc_auc'
          ,cv=cv_num
          ,random_state=42
          ,max_iter=10000
          ,fit_intercept=True
          ,solver='liblinear'
          ,tol=10
      )

  searchCV_l1.fit(X_train, Y_train)
  searchCV_l2.fit(X_train, Y_train)
  print ('Max auc_roc for l1:', searchCV_l1.scores_[1].mean(axis=0).max())
  print ('Max auc_roc for l2:', searchCV_l2.scores_[1].mean(axis=0).max())

  print("Parameters for l1 regularisations")
  print(searchCV_l1.coef_)
  print(searchCV_l1.intercept_) 
  print(searchCV_l1.scores_)

  print("Parameters for l2 regularisations")
  print(searchCV_l2.coef_)
  print(searchCV_l2.intercept_) 
  print(searchCV_l2.scores_)  


  #find predicted vallues
  Y_pred_l1 = searchCV_l1.predict(X_test)
  Y_pred_l2 = searchCV_l2.predict(X_test)
  

  #Find predicted probabilities
  Y_pred_probs_l1 = searchCV_l1.predict_proba(X_test)[:,1] 
  Y_pred_probs_l2 = searchCV_l2.predict_proba(X_test)[:,1] 

  # Accuaracy of L2/L1 models
  Accuracy_l2 = metrics.accuracy_score(y_pred=Y_pred_l2, y_true=Y_test)
  Accuracy_l1 = metrics.accuracy_score(y_pred=Y_pred_l1, y_true=Y_test)

  print("Accuarcy of Logistic model with l2 regularisation : {0}".format(Accuracy_l2))
  print("Confusion Matrix")
  Plot_confusion_matrix(Y_test, Y_pred_l2)
  print("classification Report")
  print(classification_report(Y_test, Y_pred_l2))
    
  print("Accuarcy of Logistic model with l1 regularisation : {0}".format(Accuracy_l1))
  print("Confusion Matrix")
  Plot_confusion_matrix(Y_test, Y_pred_l1)
  print("classification Report")
  print(classification_report(Y_test, Y_pred_l1))

  l2_roc_value = roc_auc_score(Y_test, Y_pred_probs_l2)
  print("l2 roc_value: {0}" .format(l2_roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_probs_l2)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("l2 threshold: {0}".format(threshold))

  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  credit_card_data_Results = credit_card_data_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'Logistic Regression with L2 Regularisation','Accuracy': Accuracy_l2,'roc_value': l2_roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  l1_roc_value = roc_auc_score(Y_test, Y_pred_probs_l1)
  print("l1 roc_value: {0}" .format(l1_roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_probs_l1)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("l1 threshold: {0}".format(threshold))

  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  credit_card_data_Results = credit_card_data_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'Logistic Regression with L1 Regularisation','Accuracy': Accuracy_l1,'roc_value': l1_roc_value,'threshold': threshold}, index=[0]),ignore_index= True)
  return credit_card_data_Results


# In[127]:


# Created a common function to fit and predict on a Random Forest model
def buildAndRunRandomForestModels(credit_card_data_Results, Methodology, X_train,Y_train, X_test, Y_test ):
  #Evaluate Random Forest model

  # Create the model with 100 trees
  RF_model = RandomForestClassifier(n_estimators=100, 
                                bootstrap = True,
                                max_features = 'sqrt', random_state=42)
  # Fit on training data
  RF_model.fit(X_train, Y_train)
  RF_test_score = RF_model.score(X_test, Y_test)
  RF_model.predict(X_test)

  print('Model Accuracy: {0}'.format(RF_test_score))


  # Actual class predictions
  rf_predictions = RF_model.predict(X_test)

  print("Confusion Matrix")
  Plot_confusion_matrix(Y_test, rf_predictions)
  print("classification Report")
  print(classification_report(Y_test, rf_predictions))

  # Probabilities for each class
  rf_probs = RF_model.predict_proba(X_test)[:, 1]

  # Calculate roc auc
  roc_value = roc_auc_score(Y_test, rf_probs)

  print("Random Forest roc_value: {0}" .format(roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(Y_test, rf_probs)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("Random Forest threshold: {0}".format(threshold))
  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  credit_card_data_Results = credit_card_data_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'Random Forest','Accuracy': RF_test_score,'roc_value': roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  return credit_card_data_Results


# In[128]:


#Lets perfrom StratifiedKFold and check the results
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_SKF_cv, X_test_SKF_cv = X.iloc[train_index], X.iloc[test_index]
    Y_train_SKF_cv, Y_test_SKF_cv = Y.iloc[train_index], Y.iloc[test_index]


# In[129]:


#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
credit_card_data_Results = buildAndRunLogisticModels(credit_card_data_Results,"StratifiedKFold Cross Validation", X_train_cv,Y_train_cv, X_test_cv, Y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )


# In[130]:


#Run Random Forest Model
print("Random Forest Model")
start_time = time.time()
credit_card_data_Results = buildAndRunRandomForestModels(credit_card_data_Results,"StratifiedKFold Cross Validation",X_train_cv,Y_train_cv, X_test_cv, Y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )


# In[131]:


credit_card_data_Results


# ### Results for cross validation with StratifiedKFold:
# Looking at the ROC value we have Random Forest has provided best results for cross validation with StratifiedKFold technique
