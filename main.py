import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class kth_arm_bandit:
    def __init__(self):
        self.maxIter = 2000
        self.arms = 10
        self.A = 0
        self.Q = np.zeros(self.arms,float)
        self.N = np.zeros(self.arms,float)
        # T array keeps track of action take and action values for avg reward and optimal actions
        self.T = np.zeros([self.maxIter,2])
        # Book states mean 0 and variance 1 for Gaussian distribution for action values
        # q_*(a) ~ N(0,1)
        self.actualVal = np.random.normal(loc=0, scale=1, size=self.arms)

    def bandit(self,epslon):
        for i in range(self.maxIter):
            tempProb = random.uniform(0,1)
            if tempProb <= (1 - epslon):
                if self.Q[np.argmax(self.Q)] == 0:
                    self.A = random.randint(0,9)
                else:
                    self.A = np.argmax(self.Q)
            else:
                self.A = random.randint(0,9)
            # R_t ~ N(q_*(a),1)
            R = np.random.normal(loc=self.actualVal[self.A], scale=1)
            self.N[self.A] += 1
            self.Q[self.A] = self.Q[self.A] + ((1/self.N[self.A])*(R-self.Q[self.A]))
            if self.A == np.argmax(self.actualVal):
                self.T[i][0] += 1
            self.T[i][1] = self.Q[self.A]

    def banditData(self,epslon,actualVal,selectData):
        for i in range(self.maxIter):
            tempProb = random.uniform(0,1)
            if tempProb <= (1 - epslon):
                if self.Q[np.argmax(self.Q)] == 0:
                    self.A = random.randint(0,9)
                else:
                    self.A = np.argmax(self.Q)
            else:
                self.A = random.randint(0,9)
            # R_t ~ N(q_*(a),1)
            R = np.random.normal(loc=self.actualVal[self.A], scale=1)
            self.N[self.A] += 1
            self.Q[self.A] = self.Q[self.A] + ((1/self.N[self.A])*(R-self.Q[self.A]))
            if self.A == np.argmax(self.actualVal):
                self.T[i][0] += 1
            self.T[i][1] = self.Q[self.A]

def bandit_machine(eplson):
    maxIter = 3000
    avgRewardDist = np.zeros([maxIter,2])

    for i in range(maxIter):
        test_arm = kth_arm_bandit()
        test_arm.bandit(eplson)
        for j in range(maxIter):
            avgRewardDist[j][0] += test_arm.T[j][0]
            avgRewardDist[j][1] += test_arm.T[j][1]
        print(i)

    for i in range(len(avgRewardDist)):
        avgRewardDist[i][0] /= maxIter
        avgRewardDist[i][1] /= maxIter
    return avgRewardDist


# Part 1


# Part 2
df = pd.read_csv('Ads_Optimisation.csv')
maxIter = df.shape[0]
dfCopy = df.transpose()
test= list(df.columns)
dfCopy["actualVals"]=df[test].sum(axis=1)
actualVals = dfCopy.to_numpy()

print(df)

print("Done")