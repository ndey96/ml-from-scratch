# Written to run inside a jupyter notebook

%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = x2
y = y2

fig, ax = plt.subplots()
theta = gradient_descent(x, y, mode='batch', eps=0.0016)
x_norm = normalize(x)
y_norm = normalize(y)
ax.plot(x_norm, theta[0] + theta[1]*x_norm, color='red')
ax.scatter(x_norm, y_norm)
