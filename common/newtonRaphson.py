import numpy as np
import matplotlib.pyplot as plt

def newtonsMethod(interval, guess, f, h, maxIterations = 10, samples = None, debug = False, relTol=1e-05, abTol=1e-08):
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
        y = f(x) if samples is None else f(samples, x)
        if debug == True: print("DEBUG: x={}, y={}".format(x, y))
        pointsList.append((x, y))
        if np.isclose(y, 0.0, rtol=relTol, atol=abTol):
            iteration += 1
            break

        x = x + h(x) if samples is None else h(samples, x)
        if x < interval[0] or x > interval[1]:
            raise ValueError("There is no solution in this interval")
        iteration += 1

    # If we got here, a root was found
    root = x
    return [(root, iteration), pointsList]
