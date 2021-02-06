from .utils import transform_tensor_to_list
import numpy as np
import torch
import torch.utils.data as data

class FedOnlineTrainer(object):

    def __init__(self, client_index, train_data_local_dict, test_data_local_dict,
                 device, args, model_trainer):

        self.args = args
        self.trainer = model_trainer
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        """
        self.train_x = np.load('../../../fedml_experiments/distributed/fedonline/train_x.npy', allow_pickle=True).item()
        self.train_y = np.load('../../../fedml_experiments/distributed/fedonline/train_y.npy', allow_pickle=True).item()
        self.train_data_local_dict = dict()
        for client_idx in range(3400):
            train_dl = data.TensorDataset(torch.tensor(self.train_x[client_idx].reshape(-1, 28,28), dtype=torch.float), torch.tensor(self.train_y[client_idx], dtype=torch.long))
            self.train_data_local_dict[client_idx] = data.DataLoader(dataset=train_dl,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                drop_last=False)
        """
        self.local_sample_number = len(self.train_data_local_dict[client_index].dataset)
        # self.local_batch_size = ?
        self.test_data_local_dict = test_data_local_dict
        self.device = device

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self):
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample