import numpy as np
import matplotlib.pyplot as plt

logLikely = lambda x, th: -np.log(np.pi) + 2 * np.log(x - th)
obs = np.array([1.77, -0.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29, 3.71, -2.40, 4.53, -0.07, -1.05, -13.87, -2.53, -1.75, 0.27, 43.21])

x = np.linspace(0, 3, 200)
plt.plot(x, [logLikely(obs, th_i) for th_i in x])
plt.show()
