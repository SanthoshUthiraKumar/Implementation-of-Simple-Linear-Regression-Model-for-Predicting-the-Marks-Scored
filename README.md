# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Importing the Standard Libraries.

2. Uploading the CSV file to a python compiler.

3. Obtaining the head,tail,y_pred,y_test of the data from the CSV file.

4. Plotting the graphs and finding MSE,MAE,RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Santhosh U
RegisterNumber:  212222240092

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
### 1. df.head()
![Output1](https://user-images.githubusercontent.com/119477975/229552888-0aac3c12-0a00-43c1-908c-2a6c43d111eb.png)

### 2. df.tail()
![Output2](https://user-images.githubusercontent.com/119477975/229552911-69df12c6-2975-474c-b1b8-0eda3baec285.png)

### 3. Array value of X
![Output3](https://user-images.githubusercontent.com/119477975/229552946-b1826705-3f92-48fb-8e53-563e0215598c.png)

### 4. Array value of Y
![Output4](https://user-images.githubusercontent.com/119477975/229552986-947ce331-13c0-4b64-8d9b-8190bd19042f.png)

### 5. Values of Y prediction
![Output5](https://user-images.githubusercontent.com/119477975/229553042-702ed24a-3eb3-425c-a879-77410c0832de.png)

### 6. Array values of Y test
![Output6](https://user-images.githubusercontent.com/119477975/229553084-8658339e-eed7-47c0-a961-b3069ab62180.png)

### 7. Training Set Graph
![Output7](https://user-images.githubusercontent.com/119477975/229553128-736b03fe-cc1a-4915-b48c-3b625cc99729.png)

### 8. Test Set Graph
![Output8](https://user-images.githubusercontent.com/119477975/229553251-9eddbb79-85c6-4858-ba71-f215bc14c8c5.png)

### 9. Values of MSE, MAE and RMSE
![Output9](https://user-images.githubusercontent.com/119477975/229553286-804bc3e7-1c09-49aa-8086-c6778cf1eb57.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
