import copy
import logging
import random
import time

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import torch
import wandb
import termplotlib as tpl

from .utils import transform_list_to_tensor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from fedml_api.data_preprocessing.MNIST.data_loader import read_data

class FedOnlineAggregator(object):

    def __init__(self, train_global, test_global, train_data_local_dict, test_data_local_dict, class_num, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer
        self.train_global = train_global
        self.test_global = test_global
        self.args = args
        self.val_global = self._generate_validation_set()
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.score_list = self.client_score()

        self.class_num = class_num
        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def plot_stats(self, data_ratio, bin_edges):
        #counts, bin_edges = np.histogram(train_selected_y, bins=10, range=(-0.5, 9.5))
        fig = tpl.figure()
        fig.hist(data_ratio, bin_edges, force_ascii=True)
        fig.show()

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_stats(self, client_idx):
        #train_x = dict()
        train_y = dict()
        train_y_ratio = dict()
        train_local_dict = self.train_data_local_dict
        for client in client_idx:
            #train_x[client] = np.vstack(train_local_dict[client][batch][0].numpy() for batch in range(len(train_local_dict[client])))
            #train_y[client] = np.hstack(train_local_dict[client][batch][1].numpy() for batch in range(len(train_local_dict[client])))
            train_y[client] = np.array([])
            for batch_idx, (x, labels) in enumerate(train_local_dict[client]):
                train_y[client] = np.hstack((train_y[client], labels))
            #train_y[client] = np.hstack(labels for batch_idx, (x, labels) in enumerate(train_local_dict[client]))
            train_y_ratio[client], bin_edges = np.histogram(train_y[client], bins=self.class_num, range=(-0.5, self.class_num-0.5))
            train_y_ratio[client] = train_y_ratio[client]/len(train_y[client])
        
        print('\n')
        #train_selected_x = np.vstack(train_x[client] for client in client_idx)
        #train_selected_y = np.hstack(train_y[client] for client in client_idx)
        #train_selected_y_ratio = np.hstack(train_y_ratio[client] for client in client_idx)

        train_selected_y_ratio = np.zeros(self.class_num)
        for client in client_idx:
            train_selected_y_ratio += train_y_ratio[client]
        self.plot_stats(train_selected_y_ratio, bin_edges)

    def client_score(self):
        train_local_dict = self.train_data_local_dict
        client_location = dict()
        client_prob = dict()
        pca_method, step, global_score_matrix, train_x_mean,\
             min_bound, quanti, reduced_dimension = self.global_score()
        matrix_not_nan_count = len(global_score_matrix[global_score_matrix != 0])
        global_prob_matrix = np.reciprocal(global_score_matrix)
        # global_prob_matrix = global_prob_matrix/np.sum(global_prob_matrix[global_prob_matrix != np.inf])*0.1
        global_prob_matrix = global_prob_matrix * 10 / matrix_not_nan_count
        
        for client in range(len(train_local_dict)):
            if client%500 == 0:
                print("calculating client score %d" %(client) )
            location = np.zeros(reduced_dimension)
            for i in range(reduced_dimension):
                location[i] = (train_x_mean[client][i] - min_bound[i])//step[i]
                if location[i] == quanti:
                    location[i] = quanti - 1
            client_location[client] = location
            client_prob[client] = global_prob_matrix[tuple(location.astype(np.int16))]
            expected_client_count = sum(client_prob.values())
        return client_prob


    def global_score(self, quanti=50, reduced_dimension=2, sample_per_client=10):
        train_local_dict = self.train_data_local_dict

        if self.args.load_data == True:
            train_x = dict()
            all_train_x = np.ones((1, 784))
            for client in range(len(train_local_dict)):
                if client%200 == 0:
                    print("collecting clients data %d" %(client) )
                train_x[client] = np.ones((1, 784))
                data_per_client = 0
                for batch_idx, (x, labels) in enumerate(train_local_dict[client]):
                    tmp = x.reshape([x.shape[0], 784])
                    train_x[client] = np.vstack((train_x[client], tmp))
                    if train_x[client].shape[0] > sample_per_client:
                        break
                train_x[client] = np.delete(train_x[client], 0, axis=0)
                all_train_x = np.vstack((all_train_x, train_x[client]))
            all_train_x = np.delete(all_train_x, 0, axis=0)
            # np.save(train_x, '../../../fedml_experiments/distributed/fedonline/train_x.npy')
            # np.save(all_train_x, '../../../fedml_experiments/distributed/fedonline/all_train_x.npy')
        else:
            train_x = np.load('../../../fedml_experiments/distributed/fedonline/train_x.npy')
            all_train_x = np.load('../../../fedml_experiments/distributed/fedonline/all_train_x.npy')

        pca = PCA(n_components=reduced_dimension)
        pca_method = pca.fit(all_train_x)
        all_train_x = pca_method.transform(all_train_x)
        region = np.zeros((quanti, quanti))
        step = list()
        max_bound = dict()
        min_bound = dict()
        for i in range(reduced_dimension):
            max_bound[i] = np.max(all_train_x[:,i])
            min_bound[i] = np.min(all_train_x[:,i])
            step.append((max_bound[i]-min_bound[i])/quanti)

        train_x_mean = dict()
        for client in range(len(train_local_dict)):
            train_x[client] = pca_method.transform(train_x[client])
            train_x_mean[client] = np.mean(train_x[client], axis=0)

            location = np.zeros(reduced_dimension)
            for i in range(reduced_dimension):
                location[i] = (train_x_mean[client][i] - min_bound[i])//step[i]
                if location[i] == quanti:
                    location[i] = quanti - 1
            region[int(location[0]), int(location[1])] += 1
        print('stage 2 finished')
        return pca_method, step, region, train_x_mean, min_bound, quanti, reduced_dimension
        

    def mnist_score(self, reduced_dimension):
        train_path="./../../../data/MNIST/train"
        test_path="./../../../data/MNIST/test"
        users, groups, train_data, test_data = read_data(train_path, test_path)

        pca = PCA(n_components=reduced_dimension)
        all_data = np.zeros(784)
        for user in users:
            Data = train_data[user]['x']
            all_data = np.vstack((all_data, Data))
        pca_method = pca.fit(all_data)
        all_data = pca_method.transform(all_data)

        nbrs = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(all_data)
        score = dict()
        client_idx = 0
        for user in users:
            user_x = pca_method.transform(train_data[user]['x'])
            #user_y = train_data[user]['y']
            #user_Data = np.hstack((user_x, np.expand_dims(user_y, axis = 1)))
            distances, indices = nbrs.kneighbors(user_x)
            score[client_idx] = np.mean(distances)
            client_idx += 1
        score = sorted(score.items(), key = lambda kv:(kv[1], kv[0]))
        score_list = list()
        for idx, dis in score:
            score_list.append(idx)
        score_list.reverse()
        print(len(score_list))
        return score_list

    def fixed_client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_indexes = []
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(500)
            s = self.score_list
            for client in range(client_num_in_total):
                if np.random.binomial(n=1, p=s[client]):
                    client_indexes.append(client)
            if len(client_indexes) > 10:
                for i in range(len(client_indexes)-10):
                    client_indexes.remove(random.choice(client_indexes))
            elif len(client_indexes) < 10:
                tmp_client_list = np.array(range(client_num_in_total))
                for i in range(10-len(client_indexes)):
                    add_client_index = np.random.choice(tmp_client_list)
                    while add_client_index in client_indexes:
                        add_client_index = np.random.choice(tmp_client_list)
                    tmp_client_list.delete(add_client_index)
                    client_indexes.append(add_client_index)
            #client_indexes = [411,34,989,529,49]
            client_indexes = np.array(client_indexes)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
            #for client_idx in range(5):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break
            
            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
            
            #metrics = self.trainer.test(self.train_data_local_dict[0], self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
