import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  
import pandas as pd

df = pd.read_csv('./bmi/Bodyfat.csv')
y = df['bodyfat']
X = df[['Age','Weight','Height']]

X_train, X_test, y_train, y_test = train_test_split(X,y)

mlp = MLPRegressor(hidden_layer_sizes=(20,40,40,40,20), max_iter=10000)  
mlp.fit(X_train, y_train)


y_mlp = mlp.predict(X_test)

plt.plot(X_train, y_train,"g.")
plt.plot(X_test, y_mlp, "b.")
plt.legend(['Data','Model MLP'])
plt.figure()
plt.plot(y_test,y_mlp,'.')
plt.show() 

