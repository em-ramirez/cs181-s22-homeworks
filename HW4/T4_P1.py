from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

x = [-3,-2,-1,0,1,2,3]
y = [1,1,-1,1,-1,1,1]
x = np.array(x)
y = np.array(y)
phi_x = np.array((x, -8/3 * x**2 + 2/3 * x**4))

colors = np.array(['g', 'b', 'r'])

plt.scatter(phi_x[0], phi_x[1], c = colors[y])
plt.axhline(y=-1, color='g', linestyle='-')
plt.xlabel(r'$\phi(x)_1$')
plt.ylabel(r'$\phi(x)_2$')
plt.legend()
plt.show()