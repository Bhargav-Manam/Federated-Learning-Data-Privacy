# MNIST Dataset

## Introduction
Splits `MNIST` dataset among `n_clients`, three methods are available:

### IID split (Default)
The dataset is shuffled and partitioned among `n_clients`

### By Labels Non-IID split
The dataset is split among `n_clients` as follows:
1. classes are grouped into `n_clusters`.
2. for each cluster `c` in `n_clusters`, samples are partitioned across clients using dirichlet distribution with parameter `alpha`.

Inspired by the split in [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440).

In order to use this mode, you should use argument `--by_labels_split`.

### Pathological Non-IID split
The dataset is split as follows:
1) sort the data by label
2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
3) assign each of the `n_clients` with `n_classes_per_client` shards

Similar to [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

In order to use this mode, you should use argument `--pathological_split`.

## Instructions
Run generate_data.py with a choice of the following arguments:

- `--n_clients`: number of clients, written as integer;
- `--frac`: fraction of the dataset to be used; default=``1.0``;
- `--iid`: (`bool`) if selected; "iid split" is used
- `--by_labels_split`: (`bool`) if selected; "by labels non-iid split" is used;
- `--pathological_split`: (`bool`) if selected; "pathological non-iid split" is used
- `--n_components`: number of mixture components, written as integer, ignored if `--by_labels_split`  is not used; default=``-1``;
- `--alpha`: parameter controlling clients dissimilarity, the smaller alpha is the more tasks are dissimilar; default=``0.5``;
- `--n_shards`: number of shards given to each client; ignored if `--pathological_split` is not used; default=``2``;
- `--seed` := seed to be used before random sampling of data, default is `1234`;

### Remarks
- In case `--pathological_split` and `--by_labels_split` are both selected, `--by_ labels_split` will be used.
- If `n_components=-1`, then `n_components` will be set to be equal to `n_classes(=10)`.
