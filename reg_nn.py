import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234567890)

"""
Bikin model untuk y = 2 + x + 0.5x^2 + Gaussian noise
"""
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.neural_network import MLPRegressor  
mlp = MLPRegressor(hidden_layer_sizes=(20,20,20), activation='tanh', solver='sgd', max_iter=10000)  
mlp.fit(X, y.ravel())

X_mlp = [[i] for i in np.linspace(-3,3,m)]
y_mlp = mlp.predict(X_mlp)

plt.plot(X,y,"g.")
plt.plot(X_mlp, y_mlp, "b.")
plt.legend(['Data','Model MLP'])
plt.show() 

