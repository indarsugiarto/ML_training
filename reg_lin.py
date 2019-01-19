import numpy as np
import matplotlib.pyplot as plt

DISPLAY_DATA = False

"""
Bikin data dummy dengan persamaan: y = 4 + 3x + Gaussian_noise
"""
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

if DISPLAY_DATA is True:
  plt.figure(0)
  plt.scatter(X,y)
  plt.show()

"""
Solusi menggunakan persamaan normal.
Menggunakan fungsi inv() dari Linear Algebra module-nya Numpy ( np.linalg ) untuk menghitung inverse dari matrix, 
dan fungsi dot() untuk perkalian matrix.
"""
X_b = np.c_[np.ones((100, 1)), X] # dibuat x0 = 1 supaya theta_0 tidak sama dengan 0
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Estimasi normal: ",theta_best[0], theta_best[1])

"""
Sekarang kita plot model-nya dengan dua end-point [0] dan [2]
"""
X_points = np.array([[0], [2]])
X_points_b = np.c_[np.ones((2, 1)), X_points] # dibuat x0 = 1 supaya theta_0 tidak 0
y_points = X_points_b.dot(theta_best)
#print(y_points)

"""
Solusi kedua menggunakan scikit-learn
"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Estimasi scikit: ", lin_reg.intercept_, lin_reg.coef_)
y_scikit = lin_reg.predict(X_points)

plt.figure(1)
plt.plot(X,y,"g.")
plt.plot(X_points, y_points, "r-")
plt.plot(X_points, y_scikit, "b-")
plt.axis([0,2,0,15])
plt.legend(["Data", "Model Normal", "Model Scikit"])
plt.title("Dengan Solusi Normal")
plt.show()



