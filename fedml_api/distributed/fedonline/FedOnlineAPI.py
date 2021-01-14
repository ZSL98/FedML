from mpi4py import MPI

from .FedOnlineAggregator import FedOnlineAggregator
from .FedOnlineTrainer import FedOnlineTrainer
from .FedOnlineClientManager import FedOnlineClientManager
from .FedOnlineServerManager import FedOnlineServerManager
from .MyModelTrainer import MyModelTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    print('initialized')
    return comm, process_id, worker_number


def FedML_FedOnline_distributed(process_id, worker_number, device, comm, model, test_global, train_data_local_dict, test_data_local_dict, args, model_trainer=None):
    if process_id == 0:
        print('server begin')
        init_server(args, device, comm, process_id, worker_number, model, test_global, train_data_local_dict, test_data_local_dict, model_trainer)
    else:
        print('client begin')
        init_client(args, device, comm, process_id, worker_number, model, train_data_local_dict, test_data_local_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, test_global, train_data_local_dict, test_data_local_dict, model_trainer):
    if model_trainer is None:
        model_trainer = MyModelTrainer(model)
        model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedOnlineAggregator(test_global, train_data_local_dict, test_data_local_dict, worker_num, device, args, model_trainer)

    # start the distributed training
    server_manager = FedOnlineServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_local_dict, test_data_local_dict, model_trainer=None):
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = MyModelTrainer(model)
        model_trainer.set_id(client_index)

    print(device)
    trainer = FedOnlineTrainer(client_index, train_data_local_dict, test_data_local_dict, device, args, model_trainer)
    client_manager = FedOnlineClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
