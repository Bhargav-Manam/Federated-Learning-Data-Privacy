
import numpy as np

import torch
from torch.utils.data import ConcatDataset, DataLoader

from skimage.metrics import structural_similarity

from matplotlib import pyplot as plt

from tqdm import tqdm


class Attacker:
    r"""Base class for Attacker.

    `Attacker` is a malicious entity that access the global model

    Attributes
    ----------
    clients_dict: Dict[int: Client]

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    logger: SummaryWriter

    Methods
    ----------
    __init__

    model_inversion_attack

    write_attack_logs

    """
    def __init__(
            self,
            clients_dict,
            logger,
            verbose=0
    ):
        """

        Parameters
        ----------
        clients_dict: Dict[int: Client]

        logger: SummaryWriter

        verbose: int

        """

        self.clients_dict = clients_dict

        self.dataset = self._gather_clients_datasets()

        self.targets = []
        for idx_ in range(len(self.dataset)):
            _, target = self.dataset[idx_]
            self.targets.append(int(target))

        self.n_classes = len(set(self.targets))

        self.verbose = verbose
        self.logger = logger

        self.imgs_recover = np.zeros((40, 112, 92))

        self.global_attack_loss = np.zeros(self.n_classes)

    def _gather_clients_datasets(self):
        """
        gather clients' local dataset for attack evaluation
        Returns
        -------
            * torch.utils.data.ConcatDataset

        """

        clients_datasets = []
        for _, client in self.clients_dict.items():
            client_dataset = client.train_loader.dataset
            clients_datasets.append(client_dataset)

        dataset = ConcatDataset(clients_datasets)

        return dataset

    def model_inversion_attack(self, model, n_rounds):
        """
        perform model inversion attack

        Parameters
        ----------

        model: nn.Module
            the model accessed by the attacker

        n_rounds : int

        """

        for label in range(self.n_classes):

            img_initial = torch.zeros(112 * 92).requires_grad_()

            for i in range(n_rounds):

                loss = 1 - torch.nn.functional.softmax(model(img_initial), dim=0)[label]

                loss.backward()

                if i == n_rounds - 1:
                    self.global_attack_loss[label] = loss

                gradient_img = img_initial.grad.data
                img_initial.data.add_(gradient_img, alpha=-0.5)

            img_recovered = img_initial.reshape(112, 92).detach().numpy()

            self.imgs_recover[label] = img_recovered

    def write_attack_logs(self, c_round, plot=False):
        """

        Parameters
        ----------
        c_round
        plot
        """
        global_attack_cost = self.global_attack_loss.sum() / self.n_classes

        global_attack_metric = 0.

        for idx_ in range(len(self.dataset)):
            image, label = self.dataset[idx_]

            image = image.cpu().detach().numpy().reshape(112, 92)
            image_recover = self.imgs_recover[int(label)]

            data_range = image.max() - image.min()

            score, _ = structural_similarity(image, image_recover, full=True, data_range=data_range)
            global_attack_metric += score

        global_attack_metric /= len(self.dataset)

        if self.verbose > 0:
            tqdm.write("+" * 50)
            tqdm.write("Attack..")
            tqdm.write(f"Attack Loss: {global_attack_cost:.6f} | Attack Metric: {global_attack_metric:.3f} |")
            tqdm.write("+" * 50)

        self.logger.add_scalar("Attack/Loss", global_attack_cost, c_round)
        self.logger.add_scalar("Attack/Metric", global_attack_metric, c_round)
        self.logger.flush()

        if plot:

            plt.rcParams['figure.figsize'] = (20, 12)
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(5, 8)
            for i in range(5):
                for j in range(8):
                    ax[i, j].imshow(self.imgs_recover[i * 8 + j])
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])

            plt.show()




