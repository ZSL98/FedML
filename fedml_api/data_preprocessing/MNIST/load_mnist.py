import logging

import random
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # if self.train:
        #     data = mnist_dataobj.train_data
        #     target = mnist_dataobj.train_labels
        # else:
        #     data = mnist_dataobj.test_data
        #     target = mnist_dataobj.test_labels

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)

def get_dataloader(train_ds, test_ds, train_bs, test_bs, vc_sample):
    train_sampler = RandomSampler(train_ds, num_samples=vc_sample, replacement=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, sampler=train_sampler, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def get_dataloader_sample(train_ds, test_ds, train_bs, test_bs, var_value, client_idx, vc_sample):
    proportions = np.load(var_value, allow_pickle=True).item()

    weights = np.array(proportions[client_idx])/sum(proportions[client_idx])
    sample_weights = [weights[label] for label in train_ds.tensors[1]]
    train_weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=vc_sample, replacement=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, sampler=train_weighted_sampler, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def partition_data(imb_factor, var_value, dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_nets)
                for j in range(n_nets):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_nets):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_nets):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1
    
    elif partition == "hetero_1":
        imb_factor = 0.5
        sample_num = dict()
        contain = dict()
        for i in range(10):
            sample_num[i] = int(5000 * (imb_factor**(i / (10 - 1.0))))
        for j in range(n_nets):
            t = random.randint(0, sum(list(sample_num.values())) - 1)
            for i, val in enumerate(list(sample_num.values())):
                t -= val
                if t < 0:
                    contain[j] = i
                    break
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
        for i in range(10):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            idx_k = np.random.choice(idx_k, sample_num[i], replace=False)
            for j in range(n_nets):
                if contain[j] == i:
                    net_dataidx_map[j] = np.random.choice(idx_k, 100, replace=False)
                    idx_k = np.delete(idx_k, np.where(idx_k==net_dataidx_map[j]))

    elif partition == "hetero_n":
        sample_num = dict()
        contain = dict()
        proportions = np.load(var_value, allow_pickle=True).item()
        for i in range(10):
            sample_num[i] = int(5000 * (imb_factor**(i / (10 - 1.0))))
        net_dataidx_map = {i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
        for i in range(10):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            idx_k = np.random.choice(idx_k, sample_num[i], replace=False)
            for j in range(n_nets):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], np.random.choice(idx_k, int(proportions[j][i]/n_nets), replace=False))
                #idx_k = np.delete(idx_k, np.where(idx_k==net_dataidx_map[j]))
                idx_k = np.setdiff1d(idx_k, net_dataidx_map[j])

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def load_partition_data_mnist(imb_factor, var_value, dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, vc_sample):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(imb_factor, var_value, dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    dl_obj = MNIST_truncated
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = dl_obj(data_dir, train=True, transform=transform, download=True)
    test_ds = dl_obj(data_dir, train=False, transform=transform, download=True)

    train_ds = data.TensorDataset(torch.tensor(train_ds.data.reshape(-1,784), dtype=torch.float), torch.tensor(train_ds.target, dtype=torch.long))
    test_ds = data.TensorDataset(torch.tensor(test_ds.data.reshape(-1,784), dtype=torch.float), torch.tensor(test_ds.target, dtype=torch.long))

    train_data_global, test_data_global = get_dataloader(train_ds, test_ds, batch_size, batch_size, vc_sample)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_sample(train_ds, test_ds, batch_size, batch_size, var_value,
                                                 client_idx, vc_sample)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num