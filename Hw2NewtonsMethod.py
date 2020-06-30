import numpy as np
import matplotlib.pyplot as plt
import newtonRaphson as nr

if __name__ == "__main__":
    gp = lambda x: (1 + (1 / x) - np.log(x))/(1 + x)**2
    gpp = lambda x: 3 + (4 / x) + (1 / x)**2 - 2 * np.log(x)

    print("Looking for the maximum value of g(x) = log(x)/(1 + x)")
    print("Initial guess of 3.0")

    hStep = lambda x: ((x+1)*(1 + (1 / x) - np.log(x)))/(3 + (4 / x) + (1 / x)**2 - 2*np.log(x))
    retList = newtonsMethod([2, 5], 3.0, gp, hStep)

    print("The root of g'(x) is: {:.5f}".format(retList[0][0]))
    print("Root found in {} iterations".format(retList[0][1]))

    print("Trying with common function")
    cmnList = nr.newtonsMethod([2,5], 3.0, gp, hStep)

    print("The root of g'(x) is: {:.5f}".format(retList[0][0]))
    print("Root found in {} iterations".format(retList[0][1]))

    # Problem 1.b: produce graph similar to that of Figure 2.3
    x = np.linspace(2.6, 4, 100)
    y = gp(x)
    
    zero = np.zeros(len(x))
    plt.plot(x, y, x, zero)
    points = retList[1]
    pointsAsArgs = ["{}, {}, 'ko'".format(x,y) for x, y in points]
    
    eval("plt.plot({}, {}, {}, {})".format(*pointsAsArgs))
    plt.show()

    raise SystemExit(0)
