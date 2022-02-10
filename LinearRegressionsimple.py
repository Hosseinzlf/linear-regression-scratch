import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("grades.csv")
y = dataset.iloc[:,-1].values
x = dataset.iloc[:,-2:-1].values
a = np.ones((50,1))
X = np.concatenate((a,x),axis=1)
theta = np.random.rand(1,2)
alpha = 0.01
for i in range(1000):
    prediction = (np.dot(theta,X.T)-y.T)
    theta = theta-(alpha/50)*(np.dot(prediction,X))
    M = theta
plt.scatter(x,y)
plt.plot(x,np.dot(X,theta.T),"r")
plt.show()
