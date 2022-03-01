import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class kth_arm_bandit:
    def __init__(self, maxIter=2000):
        self.maxIter = maxIter
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

    def banditData(self,epslon,sampleData):
        self.T = np.zeros([len(sampleData), 2])
        for i in range(len(sampleData)):
            if i == len(sampleData)-1:
                print()
            tempProb = random.uniform(0,1)
            if tempProb <= (1 - epslon):
                self.A = np.argmax(self.Q)
                if self.A == 0:
                    tmpList = np.where(self.Q == 0)
                    self.A = np.random.choice(tmpList, 1, replace=False)

            else:
                self.A = random.randint(0,9)
            # R_t ~ N(q_*(a),1)
            q_a = sampleData[i][self.A]
            R = np.random.normal(loc=q_a,scale=1)
            self.N[self.A] += 1
            self.Q[self.A] = self.Q[self.A] + ((1/self.N[self.A])*(R-self.Q[self.A]))
            if self.A == 4:
                self.T[i][0] += 1
            self.T[i][1] = self.Q[self.A]

def bandit_machine(eplson,df=pd.DataFrame()):
    maxIter = 500
    if len(df.columns) != 0:
        df = df.sample(frac=1)
        sampleData = df.to_numpy()
        maxIter = len(sampleData)
        avgRewardDist = np.zeros([maxIter,2])
        for i in range(maxIter):
            test_arm = kth_arm_bandit(len(sampleData))
            test_arm.banditData(eplson,sampleData)
            for j in range(maxIter):
                avgRewardDist[j][0] += test_arm.T[j][0]
                avgRewardDist[j][1] += test_arm.T[j][1]
            print(i)

        for i in range(len(avgRewardDist)):
            avgRewardDist[i][0] /= maxIter
            avgRewardDist[i][1] /= maxIter
    else:
        avgRewardDist = np.zeros([maxIter, 2])
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

# Part 2
df = pd.read_csv('Ads_Optimisation.csv')
bandit_epslon_0 = bandit_machine(0,df)
bandit_epslon_0_01 = bandit_machine(0.01,df)
bandit_epslon_0_1 = bandit_machine(0.1,df)
#test_arm.banditData(0.01,sampleData)
#test_arm.banditData(0.1,sampleData)

# Part 1
bandit_epslon_0 = bandit_machine(0)
bandit_epslon_0 = np.transpose(bandit_epslon_0)
bandit_epslon_0_01 = bandit_machine(0.01)
bandit_epslon_0_01 = np.transpose(bandit_epslon_0_01)
bandit_epslon_0_1 = bandit_machine(0.1)
bandit_epslon_0_1 = np.transpose(bandit_epslon_0_1)
fig, axis = plt.subplots(2)

axis[0].plot(bandit_epslon_0[0])
axis[0].plot(bandit_epslon_0_01[0])
axis[0].plot(bandit_epslon_0_1[0])
axis[0].legend(['epsilon = 0','epsilon = 0.01','epsilon = 0.1'], loc='upper left')
axis[0].set_ylim(0,1)
axis[0].set_xlim(0,2000)
axis[0].set_xlabel('Steps')
axis[0].set_ylabel('% Optimal action')

axis[1].plot(bandit_epslon_0[1])
axis[1].plot(bandit_epslon_0_01[1])
axis[1].plot(bandit_epslon_0_1[1])
axis[1].legend(['epsilon = 0','epsilon = 0.01','epsilon = 0.1'], loc='upper left')
axis[1].set_ylim(0,1.5)
axis[1].set_xlim(0,2000)
axis[1].set_xlabel('Steps')
axis[1].set_ylabel('Average Reward')

plt.show()




print("Done")