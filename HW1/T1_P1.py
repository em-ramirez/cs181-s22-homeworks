#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

from ast import arg
import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]



def compute_loss(tau):
    loss_arr = []
    for n in range(len(data)):
        f_arr = []

        # iterate through kernel-based regressor to sum the kernel function
        for m in range(len(data)):
            if m == n:
                continue
            f_arr.append(np.exp(-(data[n][0] - data[m][0]) * (data[n][0] - data[m][0]) / tau) * data[m][1])

        loss_arr.append((data[n][1] - sum(f_arr))**2)

    return sum(loss_arr)

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))



# plotting graphs for x* and the predictions f(x*) for different tau values:
x_star = np.arange(0, 12, 0.1, float)
fx_star1 = []
fx_star2 = []
fx_star3 = []
for x in x_star:
    f_arr1 = []
    f_arr2 = []
    f_arr3 = []
    for n in data:
        if x == n[0]:
            pass
        f_arr1.append(np.exp(-(x - n[0]) * (x - n[0]) / 0.01) *n[1])
        f_arr2.append(np.exp(-(x - n[0]) * (x - n[0]) / 2) *n[1])
        f_arr3.append(np.exp(-(x - n[0]) * (x - n[0]) / 100) *n[1])
    fx_star1.append(sum(f_arr1))
    fx_star2.append(sum(f_arr2))
    fx_star3.append(sum(f_arr3))

plt.plot(x_star, fx_star1, "r", label=r'$\tau = 0.01$')
plt.plot(x_star, fx_star2, "b", label=r'$\tau = 2$')
plt.plot(x_star, fx_star3, "g", label=r'$\tau = 100$')
plt.xlabel(r'$x^*$')
plt.ylabel(r'$f(x^*)$')
plt.title(r'Graphs for $f(x^*)$ and $x^*$ with different $\tau$ values')
plt.legend()
plt.show()
    