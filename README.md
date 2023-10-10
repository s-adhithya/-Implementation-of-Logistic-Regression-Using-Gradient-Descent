# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.


4. Predict the values of array.
 
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6. Obtain the graph


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Adhithya.S
RegisterNumber:  212222240003
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array values of x
![Screenshot 2023-10-10 092830](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/bf2f5884-f523-4650-a212-ae809c89e2d7)

### Array values of y
![Screenshot 2023-10-10 092839](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/99f685b7-2898-4b2b-bc8b-4a80cb1a6032)

### Exam 1 - score graph
![Screenshot 2023-10-10 092903](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/19919474-fd3b-42d0-a4fb-a154646fdf22)

### Sigmoid function graph
![Screenshot 2023-10-10 092915](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/1c56bd3a-06ac-4b86-9e66-2a155e22c38a)

### x_train_grad value
![Screenshot 2023-10-10 092923](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/bb0e6322-bd0c-45b8-bb95-2e23b6b8c19e)

### y_train_grad value
![Screenshot 2023-10-10 092934](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/2df37f8d-8bb5-4477-9063-d4e2c9ba2ba3)


### Print res.x
![Screenshot 2023-10-10 092950](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/01ebc28e-b266-4a6d-9706-49acb0925285)

### Decision boundary - graph for exam score
![Screenshot 2023-10-10 093002](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/055e4d0a-eeba-4b58-9710-de5dba1b5b65)

### Probability value
![Screenshot 2023-10-10 093016](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/c5172bc8-4182-4437-a119-cce20cd35fab)

### Prediction value of mean
![Screenshot 2023-10-10 093027](https://github.com/s-adhithya/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497423/10b555a2-3811-4b9c-8dd4-3bce5bc70708)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

