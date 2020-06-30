import numpy as np
import matplotlib.pyplot as plt
import fixedPointIteration as fpi
import newtonRaphson as nr
import bisection as bs
import secantMethod as sm

print("Problem 2.1(a): Graph the log likelihood function")

line = lambda n, scalar: np.ones(n) * scalar
zline = lambda n: line(n, 0)

# Using the Cauchy distribution given in problem 2.1 on Page 54
# X ~ Cauchy(Theta, 1)
#logLikely = lambda x, th: np.log((np.pi * (1 + (x - th)**2))**-1).sum()
logLikely = lambda x, th: -1 * (np.log(np.pi) + np.log(1 + (x - th)**2)).sum()
#logLikely = lambda x, th: -1 * (np.log(np.pi)) * np.log(1 + (x - th)**2).sum()
obs = np.array([1.77, -0.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29, 3.71, -2.40, 4.53, -0.07, -1.05, -13.87, -2.53, -1.75, 0.27, 43.21])

gp = lambda x, th: 2 * ((x - th)/((x - th)**2 + 1)).sum()
gpp = lambda x, th: 2 * (((2*(th - x)**2 / ((th - x)**2 + 1)) - 1)/((th - x)**2 + 1)).sum()
step = lambda x, th: gp(x, th)/gpp(x, th)

wantToSee = nr.newtonsMethod([-1, 0], -0.2, gp, step, maxIterations = 500, samples = obs)
print("wantToSee[0] {}".format(wantToSee[0]))

n = 1200
thetas = np.linspace(-14, 57, n)
y = np.array([logLikely(obs, t) for t in thetas])
plt.ylabel("$\ell(\\theta) = \sum \pi (1 + (x - \\theta)^2)^-1$")
plt.xlabel("$\\theta \in [-14, 57]$")
plt.title("Plot of Loglikely function")
plt.plot(thetas, y, thetas, line(n, -127), "k", zline(n), np.linspace(-180, -70, n), "k", wantToSee[0][0], logLikely(obs, wantToSee[0][0]), "ro", markersize=1)
plt.show()

# now, find estimates for theta using the Fixed-Point Iterative Method
print("Now use Newton-Raphson method to find the maximum")
intervals = [[-11, 0.5], [-1, 0.5], [0, 1], [1.5, 3], [3.0, 3.5], [4.7, 6], [7, 9], [8, 10], [38, 40], [40, 50]]
for invl in intervals:
    try:
        vals = nr.newtonsMethod(invl, invl[0], gp, step, maxIterations = 500, samples = obs, debug=True)
        print("Starting point {}; max {}; iterations {}".format(invl[0], vals[0][0], vals[0][1]))
    except:
        print("No maximum found in [{}, {}]".format(invl[0], invl[1]))

#print("Now, optimize using Bisection Method")
#for intv in [[-2, 0, -1], [0, 3, 1]]:
#    vals = bs.bisectionMethod(intv[0], intv[1], gp, maxIterations = 50, guess = intv[2], samples = obs)
#    print("Maximum {}; Iterations {}; Initial Guess {}".format(vals[0], vals[1], intv[2]))
#
#print("Now, find the maximum using Fixed-Point Iteration")
#vals = fpi.fixedPointMethod(-3, 0, gp, 1, samples = obs)
#print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 1))
#vals = fpi.fixedPointMethod(-3, 0, gp, 0.64, samples = obs)
#print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 0.64))
#vals = fpi.fixedPointMethod(-3, 0, gp, 0.25, samples = obs)
#print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 0.25))
#
#print("Now, find the maximum using Secant Method")
#vals = sm.secantMethod(-2, -1, gp, maxIterations = 50, samples = obs, debug = True)
#print("Theta {} in {} iterations".format(vals[0], vals[1]))
#vals = sm.secantMethod(-3, 3, gp, maxIterations = 50, samples = obs, debug = True)
#print("Theta {} in {} iterations".format(vals[0], vals[1]))

raise SystemExit(0)
