#import nexessary libraries for analysis
import bamboolib as bam #easy to do data manipulation and highly recommend going to https://bamboolib.8080labs.com/ and that will help you follow along
import pandas as pd #data manipulation
import matplotlib.pyplot as plt
import numpy as np

y=pd.read_csv(r'Trainings\Netflix_Regression.csv')

#data cleanup and investigation
y = y.loc[~(y['Gender'].isna())]
y['User ID'] = y['User ID'].astype('string')
y['Date'] = pd.to_datetime(y['Date'], infer_datetime_format=True)
y['Day'] = y['Day'].astype('string')
y['Show'] = y['Show'].astype('string')
y = pd.get_dummies(y, columns=['Gender'], drop_first=False, dummy_na=False)
y

#model processing

from sklearn.model_selection import train_test_split


y
#Split
#x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.20)
x = y[['Season', 'Episode', 'Time Watched', 'Time of Day', 'Gender_Female', 'Gender_Male']]
Y = y[['Completed']]

x_train, x_test, z_train, z_test = train_test_split(x, Y, test_size=0.20)

#logistic regression model

import statsmodels.api as sm

Xlog2 = sm.add_constant(x_train) 
logr_model = sm.Logit(z_train, Xlog2) 
logr_fit = logr_model.fit()
print(logr_fit.summary())

#confusion matrix for the results
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
logistic_regression= LogisticRegression()
model=logistic_regression.fit(x_train,z_train)

#view results

model.fit(x_train, z_train)
plot_confusion_matrix(logistic_regression, x_test, z_test)  
plt.show()
