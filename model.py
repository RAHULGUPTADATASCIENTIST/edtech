# -*- coding: utf-8 -*-
#importing the data
import pandas as pd
import pickle
df=pd.read_csv("C:/Users/acer/Downloads/edtech_data.csv")
df.columns
#data preprocessing and EDA
df.describe
df.head
df.shape
import numpy as np
df.dtypes
#dropping the enrolled_students column as it is not necessary for the prediction
df_new=df.drop("enrolled_students",axis=1)
#one hot encoding  and getting dummies for getting the numerical data
df_new["placement"]=pd.get_dummies(df_new.placement,drop_first=True)
 #yes=1  and  no=0  in the placement columns
df_new["course_category"]=pd.get_dummies(df_new.course_category,drop_first=True)
 #pg_course=0  and  skill_enhancement=1  in the course_category columns
df_new["course_type"]=pd.get_dummies(df_new.course_type,drop_first=True)
 #offline=0  and  online=1  in the course_category columns
# one hot encoding for the state and course title
a=pd.get_dummies(df_new.state,drop_first=True)
df_new=pd.concat([df_new,a],axis=1)
#drop state column
df_new.drop("state",axis=1,inplace=True)
#now for the course title
b=pd.get_dummies(df_new.course_title,drop_first=True)
df_new=pd.concat([df_new,b],axis=1)
#drop course_title column
df_new.drop("course_title",axis=1,inplace=True)
#EDA (exploratory data analysis)
import matplotlib.pyplot as plt
import seaborn as sns
correlation=df_new.corr()
#constructing a heat map for finding the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt=".1f",annot=True,annot_kws={'size':8},cmap="Blues")
#distribution plot
sns.distplot(df_new["price"],color="red")

#now defining the predictors and the target columns and doing the train_test split


predictors = df_new.loc[:, df_new.columns!="price"]
type(predictors)

target = df_new["price"]
type(target)
df_new.columns
# Train Test partition of the data and perfoming the random forest regressor as it has given best result in automl by pycaret
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=2)
from sklearn.ensemble import RandomForestRegressor as RR
regressor=RR(n_estimators=350,random_state=2)
regressor.fit(x_train,y_train)
RR(bootstrap=True,ccp_alpha=0.0,criterion="mse",max_depth=None,max_features="auto",max_leaf_nodes=None,max_samples=None,min_impurity_decrease=0.0,
   min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=350,n_jobs=None,oob_score=False,random_state=2,verbose=0,warm_start=False)







#predicting a new result
y_pred=regressor.predict(x_test)
print(y_pred)
## accuracy score
from sklearn import metrics
r_square=metrics.r2_score(y_test, y_pred)
print (r_square)
#plotting the actual price and the predicted price
plt.plot(y_test,color="blue",label="actual_price")
plt.plot(y_pred,color="red",label="predicted_price")
plt.title("Actual_price vs Predicted_price")
plt.xlabel("values")
plt.ylabel("price")
plt.legend()
plt.show()
#save the model to the disk
filename="model.pkl"
pickle.dump(regressor,open(filename,"wb"))
model=pickle.load(open("model.pkl","rb"))

