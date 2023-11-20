import os
import time
import json

import numpy as np
from tqdm import tqdm

from torchvision.datasets import MNIST

from constants import *

import warnings


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
        n_clients,
        frac,
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

    n_clients: int
        number of clients

    frac: fraction of dataset to use

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_clients`) of dictionaries, storing the data and metadata for each client

    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    n_samples = int(len(dataset) * frac)

    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False)
    rng.shuffle(selected_indices)

    clients_indices = iid_divide(selected_indices, n_clients)

    return clients_indices


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, rng=None):
    """
    split classification dataset among `n_clients` in a non-IID fashion. The dataset is split as follows:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution
    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param rng: random number generator; default is None
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False).tolist()

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        label = dataset.targets[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = rng.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = rng.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    return clients_indices


def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, rng=None):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards
    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
        :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param rng:random number generator; default is None
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False).tolist()

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        label = dataset.targets[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    rng.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices


def split_tasks_per_types(num_clients, joint_probability):
    """Split clients into client types according to `JOINT_PROBABILITY_MATRIX`.

    The number of clients per client type is computed proportionally to joint_probability.

    The result is given as a `numpy.array` of shape `(joint_probability.flatten(), )`. The elements
    correspond to the number of clients for each client type.

    Parameters
    ----------
    num_clients: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    Returns
    -------
        * `numpy.array` of shape `(joint_probability.flatten(), )`

    """

    count_per_task_type = (num_clients * joint_probability.flatten()).astype(int)
    remaining_clients = num_clients - count_per_task_type.sum()
    for task_type in range(count_per_task_type.shape[0]):
        if (remaining_clients > 0) and (count_per_task_type[task_type] != 0):
            count_per_task_type[task_type] += 1
            remaining_clients -= 1

    if count_per_task_type[1] != 0:
        print(f"==> {count_per_task_type[1]} MORE AVAILABLE CLIENTS are created")
    if count_per_task_type[4] != 0:
        print(f"==> {count_per_task_type[4]} LESS AVAILABLE CLIENTS are created")

    return count_per_task_type


def generate_tasks_types(
        num_clients,
        joint_probability,
        rng
):
    """Generate tasks_types types

    The tasks_types has an `availability` type, and a `stability` type;

    The `availability` and `stability` types are sampled according to `JOINT_PROBABILITY_MATRIX`

    The result is given as a `numpy.array` of shape `(num_clients, 2)`. The columns
    correspond to the availability type, and stability type, respectively.

    Parameters
    ----------
    num_clients: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    rng: `numpy.random._generator.Generator`

    Returns
    -------
        * `numpy.array` of shape `(num_clients, 2)`

    """

    assert np.abs(joint_probability.sum() - 1) < ERROR, "`joint_probability` should sum-up to 1!"

    clients_indices = rng.permutation(num_clients)
    count_per_cluster = split_tasks_per_types(num_clients, joint_probability)
    indices_per_cluster = np.split(clients_indices, np.cumsum(count_per_cluster[:-1]))
    indices_per_cluster = np.array(indices_per_cluster, dtype=object).reshape(joint_probability.shape)

    clients_types = np.zeros((num_clients, 2), dtype=np.int8)

    for availability_type_idx in range(joint_probability.shape[0]):
        indices = np.concatenate(indices_per_cluster[availability_type_idx])
        clients_types[indices, 0] = availability_type_idx

    for stability_idx in range(joint_probability.shape[1]):
        indices = np.concatenate(indices_per_cluster[:, stability_idx])
        clients_types[indices, 1] = stability_idx

    return clients_types


def compute_availability(availability_type, availability_parameter):
    """ compute stability value for given stability type and parameter

    Parameters
    ----------
    availability_type: str

    availability_parameter: float

    Returns
    -------
        * float:

    """
    if not (-0.5 + ERROR <= availability_parameter <= 0.5 - ERROR):
        warnings.warn("availability_parameter is automatically clipped to the interval (-1/2, 1/2)")
        availability_parameter = np.clip(availability_parameter, a_min=-0.5 + ERROR, a_max=0.5 - ERROR)

    if availability_type == "available":
        availability = 1 / 2 + availability_parameter

    elif availability_type == "unavailable":
        availability = 1 / 2 - availability_parameter

    else:
        error_message = ""
        raise NotImplementedError(error_message)

    return availability


def compute_stability(stability_type, stability_parameter, availability, rng=None):
    """ compute stability value for given stability type and parameter, and the value of availability

    Parameters
    ----------
    stability_type: str

    stability_parameter: float

    availability: float
        the value of the availability

    rng: random number generator; default is None

    Returns
    -------
        * float:

    """
    if not (0. <= stability_parameter <= 1.):
        warnings.warn("stability_parameter is automatically clipped to the interval (0, 1)")
        stability_parameter = np.clip(stability_parameter, a_min=0, a_max=1)

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    if stability_type == "stable":
        stability = np.max([
            stability_parameter,
            1 - 1 / availability + ERROR,
            1 - 1 / (1 - availability) + ERROR
        ])

    elif stability_type == "shifting":
        # TODO: add specific argument
        eps = 0.01

        stability = np.clip(
            0.5 * (rng.normal(loc=0.0, scale=eps) + eps),
            a_min=np.max([-1 + ERROR, 1 - 1 / availability + ERROR, 1 - 1 / (1 - availability)]) + ERROR,
            a_max=1 - ERROR
        )

    elif stability_type == "unstable":
        stability = np.max([
            -stability_parameter,
            1 - 1 / availability + ERROR,
            1 - 1 / (1 - availability) + ERROR
        ])

    else:
        error_message = f"{stability_type} is not a possible stability_type; possible are:"
        for t in STABILITY_TYPES:
            error_message += f"\"{t}\", "

        raise NotImplementedError(error_message)

    return stability


def generate_data(
        dataset,
        frac,
        split_type,
        n_classes,
        n_train_samples,
        n_clients,
        n_components,
        alpha,
        n_shards,
        availability_parameter,
        stability_parameter,
        save_dir,
        rng=None
):
    if split_type == "pathological_split":
        print(f"==> Pathological Non-IID Split")
        clients_indices = \
            pathological_non_iid_split(
                dataset,
                n_classes,
                n_clients,
                n_shards,
                frac,
                rng
            )
    elif split_type == "by_labels_split":
        print(f"==> Data are split by labels (non-IID)")
        clients_indices = \
            by_labels_non_iid_split(
                dataset,
                n_classes,
                n_clients,
                n_components,
                alpha,
                frac,
                rng
            )
    else:
        print("==> Data are split IID")
        clients_indices = \
            iid_split(
                dataset=dataset,
                n_clients=n_clients,
                frac=frac,
                rng=rng
            )

    all_clients_cfg = dict()

    clients_types = generate_tasks_types(
        num_clients=n_clients,
        joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
        rng=rng,
    )

    for client_id in tqdm(range(n_clients), total=n_clients, desc="Clients.."):
        client_indices = np.array(clients_indices[client_id])
        train_indices = client_indices[client_indices < n_train_samples]
        test_indices = client_indices[client_indices >= n_train_samples]

        train_data, train_targets = dataset.data[train_indices], dataset.targets[train_indices]
        test_data, test_targets = dataset.data[test_indices], dataset.targets[test_indices]

        client_dir = os.path.join(os.getcwd(), save_dir, f"client_{client_id}")

        save_data(
            save_dir=client_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        availability_id = clients_types[client_id][0]
        stability_id = clients_types[client_id][1]

        availability = \
            compute_availability(
                availability_type=AVAILABILITY_TYPES[availability_id],
                availability_parameter=availability_parameter
            )

        stability = \
            compute_stability(
                stability_type=STABILITY_TYPES[stability_id],
                stability_parameter=stability_parameter,
                availability=availability,
                rng=rng
            )

        all_clients_cfg[str(client_id)] = {
            "indices": client_indices.tolist(),
            "client_dir": client_dir,
            "availability_type": AVAILABILITY_TYPES[availability_id],
            "availability": availability,
            "stability_type": STABILITY_TYPES[stability_id],
            "stability": stability

        }

    return all_clients_cfg
