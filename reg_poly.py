import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234567890)

"""
Bikin model untuk y = 2 + x + 0.5x^2 + Gaussian noise
"""
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

#plt.scatter(X,y)
#plt.show()

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
-------------------
Biasanya menggunakan ekstraksi fitur:
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_poly = LinearRegression()
lin_poly.fit(X_poly, y)
print("Estimasi parameter: ", lin_poly.intercept_, lin_poly.coef_)
y_poly = lin_poly.predict(X_poly)
"""

from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((\
   ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),\
   ("sgd_reg", LinearRegression()),\
))
polynomial_regression.fit(X,y)
X_test = [[i] for i in np.linspace(-3,3,m)]
y_poly = polynomial_regression.predict(X_test)

plt.plot(X,y,"g.")
plt.plot(X_points, y_lin, "r-")
plt.plot(X_test, y_poly, "b.")
plt.legend(['Data','Model Linier','Model Polinomial'])
plt.show() 

