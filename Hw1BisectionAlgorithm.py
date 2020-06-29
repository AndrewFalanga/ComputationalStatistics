import numpy as np
import matplotlib.pyplot as plt
import math

# Home work 1: Implement the bisection algorithm from page 23
# in the Computational Statistics textbook.

# Given: the function 
#         log(x)
#    g(x)=------
#         (1+x)
#
# This yeilds
#             1+ (1/x) - log(x)
#   g'(x) = --------------------
#                 (1 + x)^2

gPrime = lambda x: (1 + (1/x) - np.log(x))/(1 + x)**2
newBound = lambda a,b: (1/2)*(a + b)

def rootOfGPrime(a, b, g, maxIterations = 10):
    # The first check, if g(a) * g(b) > 0, there is no root
    if g(a) * g(b) > 0.0:
        raise ArithmeticError("No 0 in [{}, {}]".format(a, b))

    found = False
    iteration = 0
    while iteration < maxIterations:
        # Generate a guess by means of
        #  x(t) = (1/2)*(g(a) * g(b))
        x_t = (1/2)*(a + b)

        # Ok, the checks are now done, perform the algorithm
        rootCandidate = g(x_t)

        print("x_t {} rootCandidate is {} [{}, {}]".format(x_t, rootCandidate, a, b))
        # using default values for isclose() as this defaults to a relative
        # tolerance of 1*10^-9 and an absolute tolerance of 0.0
        if np.isclose(rootCandidate, 0.0):
            found = True
            break
        
        # We haven't found a root, so shrink the interval
        if g(a) * g(x_t) <= 0:
            b = x_t
        else:
            a = x_t

        iteration += 1

    if found:
        print("root candidate found after {} iterations".format(iteration))
        return abs(rootCandidate) if rootCandidate == -0.0 else rootCandidate
    else:
        raise ArithmeticError("Unable to find a root for g in [{}, {}]".format(a, b))


if __name__ == "__main__":
    print("Looking for some roots")
    print("Starting with something simple: x^2")

    root = rootOfGPrime(-1, 1, lambda x: 2*x, 4)
    assert math.isclose(0.0, root)
    print("{} is a local minimum for x^2".format(root))

    # Find root of x^2 + x
    root = rootOfGPrime(-2, 0, lambda x: 2*x + 1, 3)
    assert math.isclose(root, 0.0)
    print("{} is a local minimum x^2 + x".format(root))

    root = rootOfGPrime(-1, 1, lambda x: -2*x)
    print("{} minimizes -x^2 - 1".format(root))

    root = rootOfGPrime(1, 5, lambda x: (1 + (1/x) - np.log(x))/(1 + x)**2, 20)
    print("{} maximizes (log(x))/(1+x) inside [1, 5]".format(root))
