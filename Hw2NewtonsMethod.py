import numpy as np
import matplotlib.pyplot as plt
import newtonRaphson as nr

# Using the same math function as before: g(x) = log(x)/(1 + x)
def newtonsMethod(interval, guess, f, h, maxIterations = 10):
    '''
    newtonsMethod - finds the root of f through estimation via evaluation of
    the tangent lines of that function.
    f - The function to evaluate, g'(x)  (must be callable)
    h - The callable version of g'/g''  (must be callable)
    interval - The interval upon which f() is continuously differentiable.
    guess - An initial starting point
    maxIterations - The number of times to iterate, or loop, looking for the root
    '''
    if not callable(f) or not callable(h):
        raise ValueError("f and/or h must be callable objects")
    if not interval[0] <= guess <= interval[1]:
       
        raise ValueError("Initial guess lies outside of the provided interval")
        
    iteration = 0

    # Contains a list of tuples which may be treated as x,y pairs
    pointsList = []
    x = guess
    while iteration < maxIterations:
        y = f(x)
        if np.isclose(y, 0.0):
            break
        elif iteration >= maxIterations:
            raise ArithmeticError("No root found for f within max iterations")

        pointsList.append((x, y))
        x = x + h(x)
        iteration += 1

    # If we got here, a root was found
    root = x
    return [(root, iteration), pointsList]


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
