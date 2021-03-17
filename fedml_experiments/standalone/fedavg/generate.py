import numpy as np
import random
import copy
import scipy.stats

class generate(object):
    def __init__(self, delta):
        self.client_num = 500
        self.imb_factor = 0.1
        self.var_ori = 0
        self.sample_num = []
        for i in range(10):
            self.sample_num.append(int(5000 * (self.imb_factor**(i / (10 - 1.0)))))
        self.proportions = {i:copy.deepcopy(self.sample_num) for i in range(self.client_num)}
        #self.proportions = {i:np.zeros(10) for i in range(self.client_num)}
        #self.proportions = np.load('./proportions/proportions_1000_1e-1_4e-2.npy', allow_pickle=True).item()
        self.save_point = copy.deepcopy(self.proportions)
        self.sum_sample_num = sum(self.sample_num)
        self.delta = delta
        #x = np.array_split(np.array(range(100)), [25, 44, 59, 70, 79, 86, 91, 95, 98])
        #x = np.array_split(np.array(range(100)), [14, 27, 39, 50, 60, 69, 78, 87, 94])
        #for i in range(10):
        #    for item in x[i]:
        #        self.proportions[item][i] = self.sum_sample_num
        #print(1)

        """
        x = np.array(self.sample_num)/sum(self.sample_num)
        e = 0
        for i in range(10):
            a = np.array([0,0,0,0,0,0,0,0,0,0])
            a[i] = 1
            #t = x[i] * scipy.stats.entropy(a, x)
            t = x[i] * sum(np.abs(a-x))
            e = e + t
        print(1)
        """
        

    def var_value(self):
        proportion_base = np.array(self.sample_num)/self.sum_sample_num
        var = dict()
        for i in range(self.client_num):
            var[i] = np.var(np.array(self.proportions[i])/self.sum_sample_num - proportion_base)
        return sum(var.values())/self.client_num

    def EMD(self):
        proportion_base = np.array(self.sample_num)/self.sum_sample_num
        emd = dict()
        for i in range(self.client_num):
            emd[i] = sum(np.abs(np.array(self.proportions[i])/self.sum_sample_num - proportion_base))
        return sum(emd.values())/self.client_num
    
    def KLD(self):
        proportion_base = np.array(self.sample_num)/self.sum_sample_num
        KL = dict()
        for i in range(self.client_num):
            KL[i] = scipy.stats.entropy(np.array(self.proportions[i])/self.sum_sample_num, proportion_base)
        return sum(KL.values())/self.client_num

    def swap(self, round):
        u1,u2 = np.random.choice(range(self.client_num), 2, replace=False)
        c1,c2 = np.random.choice(range(10), 2, replace=False)
        while (self.proportions[u1][c1] < self.delta or self.proportions[u2][c2] < self.delta):
            u1,u2 = np.random.choice(range(self.client_num), 2, replace=False)
            c1,c2 = np.random.choice(range(10), 2, replace=False)
        self.proportions[u1][c1] -= self.delta
        self.proportions[u2][c2] -= self.delta
        self.proportions[u1][c2] += self.delta
        self.proportions[u2][c1] += self.delta
        if self.var_ori > self.KLD():
            #print('rollback'+str(round))
            self.proportions = copy.deepcopy(self.save_point)
        else:
            self.var_ori = self.KLD()
            self.save_point = copy.deepcopy(self.proportions)
            print(self.KLD())
        return self.var_ori 

if __name__ == "__main__":
    g = generate(50)
    triggered = {i:False for i in range(4)}
    for i in range(500000000):
        s = g.swap(i)
        if s > 0.5 and triggered[0] == False:
            np.save('./proportions/EMD_500_1e-1_0_5.npy', g.proportions)
            triggered[0] = True
        elif s > 1 and triggered[1] == False:
            np.save('./proportions/EMD_500_1e-1_1_0.npy', g.proportions)
            triggered[1] = True
        elif s > 1.5 and triggered[2] == False:
            np.save('./proportions/EMD_500_1e-1_1_5.npy', g.proportions)
            triggered[2] = True
            break
        elif s > 2 and triggered[3] == False:
            np.save('./proportions/KLD_500_5e-1_2_0.npy', g.proportions)
            triggered[3] = True
            break
    print('End')
