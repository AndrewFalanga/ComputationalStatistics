import numpy as np

def gammaDist(x, r, l):
    return ((l**r) * x **(r-1) * np.exp(-l * x))/gamma(r)


'''
gamma: Implements the gamma function.  At present, arguments are
assumed to be inetegers
'''
def gamma(a):
    if isinstance(a, int):
        return factorial(a - 1)

    return 0

def factorial(x):
    return 1 if x == 0 else factorial(x - 1) * x
