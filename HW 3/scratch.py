import numpy as np
import pandas as pd

class generalHW:
    def __init__(self, table):
        self.df = pd.read_table(table)
        self.lameIndex = 'match eyediff nosecheekdiff variabilityratio'
        self.l = lambda i, subi: self.df[self.lameIndex][i].split()[subi]

        # Most are best cast into numpy floats, but this is the Bernoulli match
        self.getMatch = lambda i: int(self.l(i, 0))
        self.getEyeDiff = lambda i: np.float(self.l(i, 1))
        self.getNoseCheekDiff = lambda i: np.float(self.l(i, 2))
        self.getVariabilityRatio = lambda i: np.float(self.l(i, 3))

class problem1(generalHW):
    def __init__(self):
        generalHW.__init__(self, '../datasets/facerecognition.dat')

        self.y = np.array([self.getMatch(i) for i in range(self.df.shape[0])])
        self.one = np.ones(self.df.shape[0])

        # my dataset has only 760 matches so, constant probability is 760/1042
        self.p = 760.0/1042.0
        logOfProb = np.log(1 - self.p)
        self.b = np.array([-logOfProb for i in range(self.df.shape[0])])


if __name__ == "__main__":
    prob1 = problem1()

