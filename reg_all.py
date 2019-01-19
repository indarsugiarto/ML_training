import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234567890)

"""
Bikin model untuk y = 2 + x + 0.5x^2 + Gaussian noise
"""
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
Regresi linier
"""
lin_reg = LinearRegression()
lin_reg.fit(X, y)

X_points = np.array([[-3], [3]])
y_lin = lin_reg.predict(X_points)

"""
Regresi polinomial
"""
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((\
   ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),\
   ("sgd_reg", LinearRegression()),\
))
polynomial_regression.fit(X,y)
X_test = [[i] for i in np.linspace(-3,3,m)]
y_poly = polynomial_regression.predict(X_test)

"""
Regresi NN
"""
from sklearn.neural_network import MLPRegressor  
mlp = MLPRegressor(hidden_layer_sizes=(20,20,20), activation='tanh', solver='sgd', max_iter=10000)  
mlp.fit(X, y.ravel())

X_mlp = [[i] for i in np.linspace(-3,3,m)]
y_mlp = mlp.predict(X_mlp)


"""
True data
"""
y_true = list()
X_true = [[i] for i in np.linspace(-3,3,m)]
for x in X_true:
  y_true.append(2 + x[0] + 0.5*(x[0]**2))


"""
Plotting hasilnya
"""
plt.plot(X_true, y_true, "m-", linewidth=2)
plt.plot(X,y,"g.")
plt.plot(X_points, y_lin, "r-", linewidth=2)
plt.plot(X_test, y_poly, "b-", linewidth=2)
plt.plot(X_mlp, y_mlp, "c-", linewidth=2)
plt.legend(['Model Asli', 'Data Sample','Model Linier','Model Polinomial','Model NN'])
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Perbandingan Model Regresi')
plt.show() 

