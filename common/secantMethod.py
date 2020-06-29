import numpy as np

def secantUpdate(x0, x1, f, samples):
    if samples is None:
        t = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
    else:
        t = x1 - f(samples, x1)*(x1 - x0)/(f(samples, x1) - f(samples, x0))

    return x1, t

def secantMethod(x0, x1, f, maxIterations = 10, samples = None, debug = False):
    update = lambda x0, x1: x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))

    iteration = 0
    while iteration < maxIterations:
        test = f(x1) if samples is None else f(samples, x1)

        if np.isclose(test, 0):
            break

        x0, x1 = secantUpdate(x0, x1, f, samples)

        if debug:
            print("x0 {}; x1 {}; test {}".format(x0, x1, test))

    else:
        raise RuntimeError("Unable to converge to solution")

    return (x1, iteration + 1)
