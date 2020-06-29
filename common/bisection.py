import numpy as np
import matplotlib.pyplot as plt
import math

def bisectionMethod(a, b, g, maxIterations = 10, guess = None, samples = None, debug = False):
    # The first check, if g(a) * g(b) > 0, there is no root
    atA = g(a) if samples is None else g(samples, a)
    atB = g(b) if samples is None else g(samples, b)
    if debug == True: print("atA:{}; atB:{}".format(atA, atB))
    if atA * atB > 0.0:
        raise ValueError("No 0 in [{}, {}]".format(a, b))
        
    # Generate a guess by means of
    #  x(t) = (1/2)*(g(a) * g(b))
    nextX = lambda: (1/2)*(a + b)

    found = False
    iteration = 0
    x_t = nextX() if guess is None else guess
    while iteration < maxIterations:
    
        # Ok, the checks are now done, perform the algorithm
        candidateVal = g(x_t) if samples is None else g(samples, x_t)

        # using default values for isclose() as this defaults to a relative
        # tolerance of 1*10^-9 and an absolute tolerance of 0.0
        if np.isclose(candidateVal, 0.0):
            found = True
            break
        
        # We haven't found a root, so shrink the interval
        atA = g(a) if samples is None else g(samples, a)
        if atA * candidateVal <= 0:
            b = x_t
        else:
            a = x_t

        x_t = nextX()
        iteration += 1

    if found:
        # This little bit of apparent craziness makes quirks with floating point
        # representation look correct when we look at it.
        root = abs(x_t) if x_t == -0.0 else x_t
        return (root, iteration)
    else:
        raise RuntimeError("Unable to find a root for g in [{}, {}]".format(a, b))
