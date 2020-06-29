import numpy as np

'''
Pg 54, proble 2.1c: Fixed Point Method
a, b -- [a, b]
L -- the Lipschitz constant, must be [0-1)
'''
def fixedPointMethod(a, b, gp, L, samples = None, guess = None, debug = False):
    G = lambda samples, x: gp(samples, x) + x
    
    x = (1/2)*(a + b) if guess is None else guess
    iteration = 0
    contracting = True
    while contracting:
        # Generate a candidate root value
        candidate = G(samples, x)
        
        # First, check that we've fully converged
        if np.isclose(candidate, x):
            break
            
        # Check that the function is contractive
        if debug == True:
            print("a:{}; b:{}; candidate:{}; x:{}".format(a, b, candidate, x))
        if not a <= candidate <= b or not a <= x <= b:
            contracting = False
            raise RuntimeError("G(x) is no longer contractive on interval")
            
        # condition 2 for contractiveness
        temp = G(samples, candidate)
        np.abs(candidate - temp) <= L*np.abs(x - candidate)
        
        # keep trying
        x = candidate
        iteration += 1
        
    return (candidate, iteration)

if __name__ == "__main__":
    # Using the Cauchy distribution given in problem 2.1 on Page 54
    # X ~ Cauchy(Theta, 1)
    logLikely = lambda x, th: -(len(x))*np.log(np.pi) - np.sum(np.log(1+(x - th)**2))
    obs = np.array([1.77, -0.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29, 3.71, -2.40, 4.53, -0.07, -1.05, -13.87, -2.53, -1.75, 0.27, 43.21])

    
    # now, find estimates for theta using the Fixed-Point Iterative Method
    gp = lambda x, th: 2 * np.sum((x - th)/(np.pi*(1 + (x - th)**2))**2)
    gpp = lambda x, th: np.sum((2*th - 2*x)/(1 + (x - th)**2)**2)
    step = lambda x, th: gp(x, th)/gpp(x, th)
    interval = [-11, 38]
    vals = fixedPointMethod(-3, 0, gp, 1, samples = obs)
    print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 1))
    vals = fixedPointMethod(-3, 0, gp, 0.64, samples = obs)
    print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 0.64))
    vals = fixedPointMethod(-3, 0, gp, 0.25, samples = obs)
    print("Converged to {} after {} iterations: L {}".format(vals[0], vals[1], 0.25))

    raise SystemExit(0)
