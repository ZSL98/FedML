import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedavg.client import Client
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import termplotlib as tpl
from itertools import combinations, permutations
import heapq
import scipy.stats
import time


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.proportions = np.load(self.args.var_value, allow_pickle=True).item()

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_num = class_num

        self.t_1 = self.args.t_1
        self.t_2 = self.args.t_2
        self.reverse = False

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.client_s = self.client_score(args.prob_method)
        # np.save('../../../fedml_experiments/client_10c_10.npy', self.client_s)
        # self.client_s = np.load('../../../fedml_experiments/client_s_100.npy', allow_pickle=True).item()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    """
    def client_stats(self, client_idx):
        #train_x = dict()
        train_y = dict()
        train_y_ratio = dict()
        train_local_dict = self.train_data_local_dict
        for client in client_idx:
            train_y[client] = np.array([])
            for batch_idx, (x, labels) in enumerate(train_local_dict[client]):
                train_y[client] = np.hstack((train_y[client], labels))
            train_y_ratio[client], bin_edges = np.histogram(train_y[client], bins=self.class_num, range=(-0.5, self.class_num-0.5))
            train_y_ratio[client] = train_y_ratio[client]/len(train_y[client])
            
            train_y_ratio[client] = np.array(self.proportions[client])/sum(self.proportions[client])
        
        print('\n')
        train_selected_y_ratio = np.zeros(self.class_num)
        for client in client_idx:
            train_selected_y_ratio += train_y_ratio[client]
        self.plot_stats(train_selected_y_ratio, bin_edges)
    """

    def pre_train(self, t1=0, t2=0):
        proportion_base = np.ones(10)*0.1
        emd_avg = dict()
        bin_edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        for t1 in np.arange(11)*0.1:
            for t2 in np.arange(11)*0.1:
                ratio = np.zeros(10)
                self.t_1 = t1
                self.t_2 = t2
                self.client_s = self.client_score(self.args.prob_method)
                emd = dict()
                for round_idx in range(100):
                    if self.args.random == 0:
                        client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
                    elif self.args.random == 1:
                        client_indexes = self.fixed_client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
                    else:
                        client_indexes = self.KLD_client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
                    train_ratio, bin_edges = self.client_stats(client_indexes)
                    ratio = ratio + train_ratio
                    emd[round_idx] = sum(np.abs(np.array(train_ratio)/sum(train_ratio) - proportion_base))
                emd_avg[(t1,t2)] = np.mean(list(emd.values()))
                print(str(t1)+'-'+str(t2))
                self.plot_stats(ratio, bin_edges)
        print(min(emd_avg,key=emd_avg.get))
        print(emd_avg[min(emd_avg,key=emd_avg.get)])
        print(1)

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            if self.args.random == 0:
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            elif self.args.random == 1:
                client_indexes = self.fixed_client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            else:
                client_indexes = self.KLD_client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            train_ratio, bin_edges = self.client_stats(client_indexes)
            self.plot_stats(train_ratio, bin_edges)
            # logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                time1 = time.time()
                w = client.train(w_global)
                time2 = time.time()
                print(time2-time1)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def client_stats(self, client_idx):
        #train_x = dict()
        train_y = dict()
        train_y_ratio = dict()
        train_local_dict = self.train_data_local_dict
        for client in client_idx:
            train_y_ratio[client], bin_edges = np.histogram(train_local_dict[client].dataset.tensors[1],
                                                                 bins=self.class_num, range=(-0.5, self.class_num-0.5))
            #train_y_ratio[client] = train_y_ratio[client]/len(train_local_dict[client].dataset.tensors[1])

            train_y_ratio[client] = np.array(self.proportions[client])/sum(self.proportions[client])

        train_selected_y_ratio = np.zeros(self.class_num)
        for client in client_idx:
            train_selected_y_ratio += train_y_ratio[client]
        
        return train_selected_y_ratio, bin_edges

    def plot_stats(self, data_ratio, bin_edges):
        #counts, bin_edges = np.histogram(train_selected_y, bins=10, range=(-0.5, 9.5))
        fig = tpl.figure()
        fig.hist(data_ratio, bin_edges, force_ascii=True)
        fig.show()
        print('\n')

    def client_score(self, prob_method):
        client_prob = dict()
        train_local_dict = self.train_data_local_dict

        if prob_method == 'y_max#1':
            category_prob = dict()
            sample_num = dict()
            train_y_num = dict()
            train_all_y_num = np.zeros(self.class_num)

            for client in range(self.args.client_num_in_total):
                #train_y_num[client], bin_edges = np.histogram(train_local_dict[client].dataset.tensors[1],
                #                                                 bins=self.class_num, range=(-0.5, self.class_num-0.5))

                train_y_num[client] = np.array(self.proportions[client])/sum(self.proportions[client])
                train_all_y_num += train_y_num[client]

            for i in range(self.class_num):
                category_prob[i] = sum(train_all_y_num)/train_all_y_num[i]/10*self.args.client_num_per_round/self.args.client_num_in_total

            for client in range(self.args.client_num_in_total):
                client_prob[client] = category_prob[np.where(np.array(train_y_num[client])==max(train_y_num[client]))[0][0]]
            return client_prob

        elif prob_method == 'y_max#2':
            category_count = dict()
            category_prob = dict()
            train_y_num = dict()

            for i in combinations(range(10), 2):
                category_count[i] = 0
            
            max_index_list = dict()
            for client in range(self.args.client_num_in_total):
                #train_y_num[client], bin_edges = np.histogram(train_local_dict[client].dataset.tensors[1],
                #                                             bins=self.class_num, range=(-0.5, self.class_num-0.5))
                train_y_num[client] = np.array(self.proportions[client])/sum(self.proportions[client])
                t = copy.deepcopy(list(train_y_num[client]))
                max_index = []
                for _ in range(2):
                    number = max(t)
                    index = t.index(number)
                    t[index] = -1
                    max_index.append(index)
                t = []
                max_index_list[client] = tuple(sorted(max_index))
                category_count[max_index_list[client]] = category_count[max_index_list[client]] + 1

            count_zero = 0
            for i in combinations(range(10), 2):
                if category_count[i] == 0:
                    count_zero = count_zero + 1

            for i in combinations(range(10), 2):
                if category_count[i] != 0:
                    category_prob[i] = min(1, 1/category_count[i]/(45-count_zero)*self.args.client_num_per_round)
            for client in range(self.args.client_num_in_total):
                client_prob[client] = category_prob[max_index_list[client]]
            return client_prob

        elif prob_method == 'y_max_mixed':
            category_count = dict()
            category_prob = dict()
            train_y_ratio = dict()

            for i in combinations(range(10), 2):
                category_count[i] = 0
            for i in range(10):
                category_count[i] = 0
            category_count['a'] = 0

            client_category = dict()
            for client in range(self.args.client_num_in_total):
                #train_y_ratio[client], bin_edges = np.histogram(train_local_dict[client].dataset.tensors[1],
                #                                                 bins=self.class_num, range=(-0.5, self.class_num-0.5))
                #train_y_ratio[client] = train_y_ratio[client]/len(train_local_dict[client].dataset.tensors[1])
                train_y_ratio[client] = np.array(self.proportions[client])/sum(self.proportions[client])
                t = copy.deepcopy(list(train_y_ratio[client]))
                max_number = []
                max_index = []
                for _ in range(2):
                    number = max(t)
                    index = t.index(number)
                    t[index] = -1
                    max_number.append(number)
                    max_index.append(index)
                t = []
                if not self.reverse:
                    if max_number[0] >= self.t_1:
                        client_category[client] = max_index[0]
                    elif max_number[0] >= self.t_2 and max_number[1] >= self.t_2:
                        client_category[client] = tuple(sorted(max_index))
                    else:
                        client_category[client] = 'a'
                else:
                    if max_number[0] >= self.t_2 and max_number[1] >= self.t_2:
                        client_category[client] = tuple(sorted(max_index))
                    elif max_number[0] >= self.t_1:
                        client_category[client] = max_index[0]
                    else:
                        client_category[client] = 'a'                    
                category_count[client_category[client]] = category_count[client_category[client]] + 1

            count_zero = list(category_count.values()).count(0)
            for i in combinations(range(10), 2):
                if category_count[i] != 0:
                    category_prob[i] = min(1, 1/category_count[i]/(56-count_zero)*self.args.client_num_per_round)
            for i in range(10):
                if category_count[i] != 0:
                    category_prob[i] = min(1, 1/category_count[i]/(56-count_zero)*self.args.client_num_per_round)
            if category_count['a'] != 0:
                category_prob['a'] = min(1, 1/category_count['a']/(56-count_zero)*self.args.client_num_per_round)

            for client in range(self.args.client_num_in_total):
                client_prob[client] = category_prob[client_category[client]]
            return client_prob

        elif prob_method == 'y_based_pca':
            client_location = dict()
            pca_method, step, global_score_matrix, position,\
                min_bound, reduced_dimension, location = self.global_score(prob_method)

        elif prob_method == 'x_based_pca':
            client_location = dict()
            pca_method, step, global_score_matrix, position,\
                min_bound, reduced_dimension, location = self.global_score(prob_method)

        matrix_not_nan_count = len(global_score_matrix[global_score_matrix != 0])
        global_prob_matrix = np.reciprocal(global_score_matrix)
        global_prob_matrix = global_prob_matrix * self.args.client_num_per_round / matrix_not_nan_count
        for client in range(len(train_local_dict)):
            if client%500 == 0:
                print("calculating client score %d" %(client) )

            location = np.zeros(reduced_dimension)
            for i in range(reduced_dimension):
                location[i] = (position[client][i] - min_bound[i])//step[i]
                if location[i] == self.args.quanti:
                    location[i] = self.args.quanti - 1
            client_location[client] = location
            if global_score_matrix[tuple(location.astype(np.int16))] >= 2:
                client_prob[client] = global_prob_matrix[tuple(location.astype(np.int16))]
            else:
                client_prob[client] = self.args.client_num_per_round/len(train_local_dict)
            expected_client_count = sum(client_prob.values())
        return client_prob


    def global_score(self, prob_method, reduced_dimension=2):
        train_local_dict = self.train_data_local_dict
        if prob_method == 'x_based_pca':
            train_x = dict()
            for client in range(len(train_local_dict)):
                local_data_num = train_local_dict[client].dataset.tensors[0].shape[0]
                train_x[client] = train_local_dict[client].dataset.tensors[0].view(local_data_num, -1)
                if client == 0:
                    all_train_x = train_x[client]
                else:
                    all_train_x = torch.cat((all_train_x, train_x[client]), 0)

            if self.args.dataset == 'fed_shakespeare':
                for client in range(len(train_local_dict)):
                    num_dict = {}
                    for i in range(90):
                        num_dict[i] = 0
                    for line in range(train_x[client].shape[0]):
                        for item in range(train_x[client].shape[1]):
                           num_dict[train_x[client][line][item].numpy().tolist()] += 1
                    train_x[client] = torch.Tensor(list(num_dict.values()))
                    train_x[client] = (train_x[client]/sum(train_x[client])).unsqueeze(0)
                    if client == 0:
                        all_train_x = train_x[client]
                    else:
                        all_train_x = torch.cat((all_train_x, train_x[client]), 0)

            pca_x = PCA(n_components=reduced_dimension)
            pca_method_x = pca_x.fit(all_train_x)
            pca_method = pca_method_x
            all_train_x = pca_method_x.transform(all_train_x)
            train_x_mean = dict()
            for client in range(len(train_local_dict)):
                train_x[client] = pca_method_x.transform(train_x[client])
                train_x_mean[client] = list(np.mean(train_x[client], axis=0))
            position = list(train_x_mean.values())

        else:
            train_y = dict()
            train_y_ratio = dict()
            for client in range(len(train_local_dict)):
                local_data_num = train_local_dict[client].dataset.tensors[0].shape[0]
                train_y[client] = train_local_dict[client].dataset.tensors[1]
                train_y_ratio[client], bin_edges = np.histogram(train_y[client], bins=self.class_num, range=(-0.5, self.class_num-0.5))
                train_y_ratio[client] = torch.Tensor(train_y_ratio[client])
                if client == 0:
                    all_train_y = train_y_ratio[client].unsqueeze(0)
                else:
                    all_train_y = torch.cat((all_train_y, train_y_ratio[client].unsqueeze(0)), 0)
            
            pca_y = PCA(n_components=reduced_dimension)
            pca_method_y = pca_y.fit(all_train_y)
            pca_method = pca_method_y
            position = pca_method_y.transform(all_train_y)

        step = list()
        max_bound = dict()
        min_bound = dict()
        region = np.zeros((self.args.quanti, self.args.quanti))
        for i in range(reduced_dimension):
            max_bound[i] = np.max(position, axis=0)[i]
            min_bound[i] = np.min(position, axis=0)[i]
            step.append((max_bound[i]-min_bound[i])/self.args.quanti)

        location = dict()
        for client in range(len(train_local_dict)):
            location[client] = np.zeros(reduced_dimension)
            for i in range(reduced_dimension):
                location[client][i] = (position[client][i] - min_bound[i])//step[i]
                if location[client][i] == self.args.quanti:
                    location[client][i] = self.args.quanti - 1
            region[int(location[client][0]), int(location[client][1])] += 1
        
        _label = dict()
        for client in range(len(train_local_dict)):
            _label[client] = np.where(np.array(train_y_ratio[client])==max(train_y_ratio[client]).tolist())[0][0]
        plt.scatter(position[:,0], position[:,1], c=list(_label.values()))
        #sns.heatmap(region)
        plt.savefig('./global_count_y.png')
        plt.close()
        return pca_method, step, region, position, min_bound, reduced_dimension, location


    def draw_client_matrix(self, round_idx, client_indexes):
        client_matrix = np.zeros((self.args.quanti, self.args.quanti)) 
        pca_method, step, global_score_matrix, position,\
             min_bound, reduced_dimension, location = self.global_score()
        for client in client_indexes:
            client_matrix[int(location[client][0]), int(location[client][1])] += 1
            self.client_all_matrix[int(location[client][0]), int(location[client][1])] += 1
        sns.heatmap(client_matrix)
        plt.savefig('./figures_1/'+str(round_idx)+'.png')
        plt.close()
        sns.heatmap(self.client_all_matrix)
        plt.savefig('./figures_all_stages_1/'+str(round_idx)+'.png')
        plt.close()

    def fixed_client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_indexes = []
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            s = self.client_s

            for client in range(client_num_in_total):
                # if np.random.binomial(n=1, p=s[self.train_data_local_dict[client].dataset.target[0]]):
                if np.random.binomial(n=1, p=min(1,s[client])):
                    client_indexes.append(client)
            if len(client_indexes) > client_num_per_round:
                # print('case1')
                for i in range(len(client_indexes)-client_num_per_round):
                    client_indexes.remove(random.choice(client_indexes))
            elif len(client_indexes) < client_num_per_round:
                # print('case2')
                tmp_client_list = np.array(range(client_num_in_total))
                for i in range(client_num_per_round-len(client_indexes)):
                    add_client_index = np.random.choice(tmp_client_list)
                    while add_client_index in client_indexes:
                        add_client_index = np.random.choice(tmp_client_list)
                    tmp_client_list = np.delete(tmp_client_list, np.where(tmp_client_list == add_client_index))
                    client_indexes.append(add_client_index)
            client_indexes = np.array(client_indexes)
        # self.draw_client_matrix(round_idx, client_indexes)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def KLD_client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        #train_local_dict = self.train_data_local_dict
        client_indexes = []
        train_y_ratio = dict()
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            for client in range(client_num_in_total):
                #train_y_ratio[client], bin_edges = np.histogram(train_local_dict[client].dataset.tensors[1],
                #                                    bins=self.class_num, range=(-0.5, self.class_num-0.5))
                #train_y_ratio[client] = train_y_ratio[client]/len(train_local_dict[client].dataset.tensors[1])
                train_y_ratio[client] = np.array(self.proportions[client])/sum(self.proportions[client])
            s_client = [client_index for client_index in range(client_num_in_total)]
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_first = np.random.choice(range(client_num_in_total), 1)[0]
            s_client.remove(client_first)
            client_indexes.append(client_first)

            for item in range(num_clients-1):
                KL_min = float('inf')
                for client in s_client:
                    P_m = sum(train_y_ratio[c] for c in client_indexes)
                    P_i = train_y_ratio[client]
                    KL = scipy.stats.entropy((P_m + P_i)/(len(client_indexes) + 1), np.ones(10)*0.1)
                    if KL < KL_min:
                        client_newly_selected = client
                        KL_min = KL
                client_indexes.append(client_newly_selected)
                s_client.remove(client_newly_selected)
            client_indexes = np.array(client_indexes)
        
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
            
    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        '''
        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        '''

        test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
        test_acc = test_metrics['test_correct'] / test_metrics['test_total']
        test_loss = test_metrics['test_loss'] / test_metrics['test_total']

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)



    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)