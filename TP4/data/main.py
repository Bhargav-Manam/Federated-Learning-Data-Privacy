"""Downloads a dataset and generates configuration file for federated simulation

Split a classification dataset, e.g., `MNIST`, among `n_clients`.

The following splitting strategies is available: `iid_split`, `by_labels_split`, `pathological_split`.

Default usage is ''iid_split'

"""
import argparse
import warnings

from utils import *
from constants import *


def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset_name",
        help="name of dataset to use, possible are {'mnist', 'faces'}",
        required=True,
        type=str
    )
    parser.add_argument(
        "--n_clients",
        help="number of clients",
        required=True,
        type=int
    )
    parser.add_argument(
        '--frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--iid",
        help="if selected, data are split iid",
        action='store_true'
    )
    parser.add_argument(
        "--by_labels_split",
        help="if selected, data are split non-iid (by labels)",
        action='store_true'
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; ignored if `--by_labels_split` is not used; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling clients dissimilarity, the smaller alpha is the more clients are dissimilar;'
             'ignored if `--by_labels_split` is not used; default is 0.5',
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--pathological_split",
        help="if selected, data are split non-iid (pathological)",
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help='path of the directory to save data and configuration;'
             'the directory will be created if not already created;'
             'if not specified the data is saved to "./{dataset_name}";',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--seed',
        help='seed for the random number generator;'
             'if not specified the system clock is used to generate the seed;',
        type=int,
        default=argparse.SUPPRESS,
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args_ = parse_arguments()

    seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed)

    if "save_dir" in args_:
        save_dir = args_.save_dir
    else:
        save_dir = os.path.join(".", args_.dataset_name)
        warnings.warn(f"'--save_dir' is not specified, results are saved to {save_dir}!", RuntimeWarning)

    os.makedirs(save_dir, exist_ok=True)

    if args_.dataset_name == "mnist" or args_.dataset_name == "faces":
        if args_.pathological_split:
            split_type = "pathological_split"
        elif args_.by_labels_split:
            split_type = "by_labels_split"
        else:
            split_type = "iid"

        dataset = get_dataset(
            dataset_name=args_.dataset_name,
            raw_data_path=os.path.join(save_dir, "raw_data")
        )

        all_clients_cfg = generate_data(
            dataset=dataset,
            frac=args_.frac,
            split_type=split_type,
            n_classes=N_CLASSES[args_.dataset_name],
            n_train_samples=N_TRAIN_SAMPLES[args_.dataset_name],
            n_clients=args_.n_clients,
            n_components=args_.n_components,
            alpha=args_.alpha,
            n_shards=args_.n_shards,
            save_dir=os.path.join(save_dir, "all_clients"),
            rng=rng
        )
    else:
        error_message = f"{args_.dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += f" {n},"
        error_message = error_message[:-1]

        raise NotImplementedError(error_message)

    save_cfg(save_path=os.path.join(save_dir, "cfg.json"), cfg=all_clients_cfg)
