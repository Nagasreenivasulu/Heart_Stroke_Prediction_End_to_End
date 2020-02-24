# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:46:48 2019

@author: 91986
"""

#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
#Reading dataset
hsdata = pd.read_csv("dataset.csv")
hsdata.head()
#Finding missing values
hsdata.info()
#Finding missing values using isnull()
hsdata.isnull()
#Finding missing values using isnull()
hsdata.isnull().sum()
hsdata = hsdata.drop(['id'],axis=1)
hsdata.head()
#Missing values are present in bmi and smoking_status
#Handling missing values
#bmi is a numerical value so I am imputing the bmi missing values with mean
hsdata['bmi'].fillna(hsdata['bmi'].mean(), inplace=True)
#Smoking status is a categorical data so I am imputing missing values of this column with mode
hsdata['smoking_status'].fillna(hsdata['smoking_status'].mode()[0], inplace=True)
hsdata.isnull().sum()
#Handling categorical data
#gender, ever_married, work_type, Residence_type and smoking_status are having categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_gender = LabelEncoder()
hsdata['gender']=labelEncoder_gender.fit_transform(hsdata['gender'])
labelEncoder_maried = LabelEncoder()
hsdata['ever_married']=labelEncoder_maried.fit_transform(hsdata['ever_married'])
labelEncoder_recidence = LabelEncoder()
hsdata['Residence_type']=labelEncoder_recidence.fit_transform(hsdata['Residence_type'])
hsdata['smoking_status'].unique()
#Creating dummy variables
dummies_worktype = pd.get_dummies(hsdata.work_type)
dummies_smoking_status = pd.get_dummies(hsdata.smoking_status)
#Merging dummies to data frame
merged = pd.concat([hsdata,dummies_worktype],axis='columns')
merged = merged.drop(['work_type','children'],axis='columns')
merged = pd.concat([merged,dummies_smoking_status],axis='columns')
final = merged.drop(['smoking_status','smokes'],axis='columns')
final.head()
final.columns
final = final[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','Residence_type', 'avg_glucose_level', 'bmi', 'Govt_job','Never_worked', 'Private', 'Self-employed', 'formerly smoked','never smoked', 'stroke']]
final.columns
final.corr(method ='pearson')
def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sn.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=0.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();  
correlation_heatmap(final)
#Age and ever_married are having correlation so I am dropping ever_married from dataset
final = final.drop(['ever_married'], axis=1)
#Finding outliers bmi
sn.boxplot(x=final['bmi'])
from scipy import stats
z = np.abs(stats.zscore(final))
print(z)
threshold = 3
print(np.where(z > 3))
print(z[1][2])
final_o = final[(z < 3).all(axis=1)]
print(final.shape)
print(final_o.shape)
x = final.iloc[:, :-1].values
y = final.iloc[:, -1].values
print(sum(y))
#Normalization
'''from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
print(x)'''
#Importing package for splitting data into training and test sets
from sklearn.model_selection import train_test_split
#Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#Dependent varibale is having large number of 0s than 1s so, the dataset is imbalanced
#dataset set is imbalanced
#Handling imbalanced dataset using SMOTE algorithm
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 1) 
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())
#Now the data is balanced
#Applying logistic regression model
from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression()  
log_regr.fit(x_train_res, y_train_res)
y_pred = log_regr.predict(x_test)
print(y_pred)
#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
#Confusion matrix visualization
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
#Saving model to disk
pickle.dump(log_regr, open('model.pkl','wb'))
#Loading model to compare
model = pickle.load(open('model.pkl','rb'))