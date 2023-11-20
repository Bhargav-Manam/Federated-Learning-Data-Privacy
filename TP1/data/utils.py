import os
import time
import json
import warnings

import numpy as np
from tqdm import tqdm

from torchvision.datasets import MNIST

from constants import *


def save_cfg(save_path, cfg):
    with open(save_path, "w") as f:
        json.dump(cfg, f)


def save_data(save_dir, train_data, train_targets, test_data, test_targets):
    """save data and targets as `.npy` files

    Parameters
    ----------
    save_dir: str
        directory to save data; it will be created it it does not exist

    train_data: numpy.array

    train_targets: numpy.array

    test_data: numpy.array

    test_targets: numpy.array

    """
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "train_data.npy"), "wb") as f:
        np.save(f, train_data)

    with open(os.path.join(save_dir, "train_targets.npy"), "wb") as f:
        np.save(f, train_targets)

    with open(os.path.join(save_dir, "test_data.npy"), "wb") as f:
        np.save(f, test_data)

    with open(os.path.join(save_dir, "test_targets.npy"), "wb") as f:
        np.save(f, test_targets)


def get_dataset(dataset_name, raw_data_path):

    if dataset_name == "mnist":

        dataset = MNIST(root=raw_data_path, download=True, train=True)
        test_dataset = MNIST(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    else:
        error_message = f"{dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += n + ",\t"

        raise NotImplementedError(error_message)

    return dataset


def iid_divide(l_, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py

    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups

    """
    num_elems = len(l_)
    group_size = int(len(l_) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l_[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l_[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l_, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l_[current_index: index])
        current_index = index

    return res


def iid_split(
        dataset,
        n_train_samples,
        n_tasks,
        save_dir,
        rng=None
):
    """
    split classification dataset among `n_clients` in an IID fashion. The dataset is split as follows:
        1) The dataset is shuffled and partitioned among n_clients

    Parameters
    ----------
    dataset: torch.utils.Dataset
        a classification dataset;
         expected to have attributes `data` and `targets` storing `numpy.array` objects

    n_train_samples: int
        number of training samples

    n_tasks: int
        number of tasks

    save_dir: str
        directory to save data for all tasks

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the data and metadata for each client

    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    n_samples = len(dataset)

    tasks_indices = iid_divide(rng.permutation(n_samples), n_tasks)

    all_tasks_cfg = dict()

    for task_id in tqdm(range(n_tasks), total=n_tasks, desc="Tasks.."):

        client_indices = tasks_indices[task_id]
        train_indices = client_indices[client_indices < n_train_samples]
        test_indices = client_indices[client_indices >= n_train_samples]

        train_data, train_targets = dataset.data[train_indices], dataset.targets[train_indices]
        test_data, test_targets = dataset.data[test_indices], dataset.targets[test_indices]

        task_dir = os.path.join(os.getcwd(), save_dir, f"task_{task_id}")

        save_data(
            save_dir=task_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        all_tasks_cfg[str(task_id)] = {
            "indices": client_indices.tolist(),
            "task_dir": task_dir
        }

    return all_tasks_cfg


def generate_data(
        dataset,
        split_type,
        n_train_samples,
        n_tasks,
        save_dir,
        rng=None
):
    if split_type == "iid":
        all_tasks_cfg = iid_split(
            dataset=dataset,
            n_train_samples=n_train_samples,
            n_tasks=n_tasks,
            save_dir=save_dir,
            rng=rng
        )
    else:
        error_message = "only `iid` is available at the moment!!" \
                        "Please pass '--iid'"

        raise NotImplementedError(error_message)

    return all_tasks_cfg
