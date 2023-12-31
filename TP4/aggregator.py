import time
import random

from abc import ABC, abstractmethod

import numpy as np

import torch
from utils.torch_utils import *

from tqdm import tqdm


class Aggregator(ABC):
    r"""Base class for Aggregator.

    `Aggregator` dictates communications between clients_dict

    Attributes
    ----------
    clients_dict: Dict[int: Client]

    clients_weights_dict: Dict[int: Client]

    global_trainer: List[Trainer]

    n_clients:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    sampling_rate: proportion of clients used at each round; default is `1.`

    n_clients_per_round:

    sampled_clients:

    logger: SummaryWriter

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__

    mix

    update_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients_dict,
            clients_weights_dict,
            global_trainer,
            sampling_rate,
            logger,
            verbose=0,
            seed=None
    ):
        """

        Parameters
        ----------
        clients_dict: Dict[int: Client]

        clients_weights_dict: Dict[int: Client]

        global_trainer: Trainer

        logger: SummaryWriter

        verbose: int

        sampling_rate: int

        seed: int

        """
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.clients_dict = clients_dict
        self.n_clients = len(clients_dict)

        self.clients_weights = []
        self.clients_weights_dict = clients_weights_dict
        for idx in range(self.n_clients):
            self.clients_weights.append(self.clients_weights_dict[idx])

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device

        self.verbose = verbose
        self.logger = logger

        self.model_dim = self.global_trainer.model_dim

        self.sampling_rate = sampling_rate
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.c_round = 0

    @abstractmethod
    def mix(self):
        """mix sampled clients according to weights

        Parameters
        ----------

        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def update_clients(self):
        """
        send the new global model to the clients
        """
        pass

    def write_logs(self):
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        for client_id, client in self.clients_dict.items():

            train_loss, train_metric, test_loss, test_metric = client.write_logs(counter=self.c_round)

            if self.verbose > 1:

                tqdm.write("*" * 30)
                tqdm.write(f"Client {client_id}..")

                tqdm.write(f"Train Loss: {train_loss:.3f} | Train Metric: {train_metric :.3f}|", end="")
                tqdm.write(f"Test Loss: {test_loss:.3f} | Test Metric: {test_metric:.3f} |")

                tqdm.write("*" * 30)

            global_train_loss += self.clients_weights_dict[client_id] * train_loss
            global_train_metric += self.clients_weights_dict[client_id] * train_metric
            global_test_loss += self.clients_weights_dict[client_id] * test_loss
            global_test_metric += self.clients_weights_dict[client_id] * test_metric

        if self.verbose > 0:

            tqdm.write("+" * 50)
            tqdm.write("Global..")
            tqdm.write(f"Train Loss: {global_train_loss:.3f} | Train Metric: {global_train_metric:.3f} |", end="")
            tqdm.write(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_metric:.3f} |")
            tqdm.write("+" * 50)

        self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
        self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
        self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
        self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)
        self.logger.flush()

    def sample_clients(self):
        """
        sample a list of clients
        """

        self.sampled_clients_ids = self.rng.sample(range(self.n_clients), k=self.n_clients_per_round)

        self.sampled_clients = [self.clients_dict[id_] for id_ in self.sampled_clients_ids]


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.

     Clients get fully synchronized with the average client.

    """
    def mix(self):

        self.sample_clients()

        clients_weights = torch.tensor(self.clients_weights, dtype=torch.float32)

        for client in self.sampled_clients:
            client.step()

        trainers_deltas = [client.trainer - self.global_trainer for client in self.sampled_clients]

        self.global_trainer.optimizer.zero_grad()

        average_models(
            trainers_deltas,
            target_trainer=self.global_trainer,
            weights=clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=False,
            average_gradients=True
        )

        self.global_trainer.optimizer.step()

        # assign the updated model to all clients_dict
        self.update_clients()

        self.c_round += 1

    def update_clients(self):
        for client_id, client in self.clients_dict.items():

            copy_model(client.trainer.model, self.global_trainer.model)
