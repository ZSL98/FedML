import numpy as np
import random
import copy
import scipy.stats
import multiprocessing


def EMD(proportions, proportion_base, client_num, sum_sample_num):
    emd = []
    for i in range(client_num):
        emd.append(sum(np.abs(np.array(proportions[i])/sum_sample_num - proportion_base)))
    return sum(emd)/client_num


def sub_process(share_var, share_dict, triggered, share_lock, proportion_base, client_num, sum_sample_num):
    delta = 500
    for i in range(500000000):
        proportions = dict()
        for j in range(client_num):
            proportions[j] = share_dict[j]
        u1,u2 = np.random.choice(range(client_num), 2, replace=False)
        c1,c2 = np.random.choice(range(10), 2, replace=False)
        while (proportions[u1][c1] < delta or proportions[u2][c2] < delta):
            u1,u2 = np.random.choice(range(client_num), 2, replace=False)
            c1,c2 = np.random.choice(range(10), 2, replace=False)
        proportions[u1][c1] -= delta
        proportions[u2][c2] -= delta
        proportions[u1][c2] += delta
        proportions[u2][c1] += delta
        emd = EMD(proportions, proportion_base, client_num, sum_sample_num)
        if emd > share_var.value:
            share_lock.acquire()
            print(emd)
            share_var.value = emd
            for k in range(client_num):
                share_dict[k] = proportions[k]
            if emd > 1.5 and triggered[2] == False:
                np.save('./proportions/emd_1000_1e-1_1_5.npy', proportions)
                triggered[2] = True
                break
            share_lock.release()
 
def main_process(proportions, proportion_base, client_num, sum_sample_num):
    share_dict = multiprocessing.Manager().dict()
    share_lock = multiprocessing.Manager().Lock()
    share_var = multiprocessing.Manager().Value('f', 0.0)
    triggered = multiprocessing.Manager().list()
    for i in range(3):
        triggered.append(False)
    process_list = []

    for i in range(client_num):
        share_dict[i] = proportions[i]

    for i in range(20):
        tmp_process = multiprocessing.Process(target=sub_process, args=(share_var, share_dict, triggered, share_lock, proportion_base, client_num, sum_sample_num))
        process_list.append(tmp_process)
 
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

if __name__ == "__main__":
    sample_num = []
    client_num = 1000
    imb_factor = 0.1
    for i in range(10):
        sample_num.append(int(5000 * (imb_factor**(i / (10 - 1.0)))))
    #proportions = {i:copy.deepcopy(sample_num) for i in range(client_num)}
    proportions = np.load('./proportions/EMD_1000_1e-1_0_5.npy', allow_pickle=True).item()
    sum_sample_num = sum(sample_num)
    proportion_base = np.array(sample_num)/sum_sample_num

    main_process(proportions, proportion_base, client_num, sum_sample_num)
