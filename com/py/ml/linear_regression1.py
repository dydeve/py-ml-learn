# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

print()

X_new = np.array([[0], [2]])
y_new = lin_reg.predict(X_new)
print(y_new)


plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-")
plt.show()
