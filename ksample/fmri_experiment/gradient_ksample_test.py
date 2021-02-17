import numpy as np
from pathlib import Path
import os
import re
import h5py
import pandas as pd
import pickle

from tqdm import tqdm
import time
from hyppo.independence import Dcorr
from sklearn.metrics import pairwise_distances
import argparse
from hyppo.tools.common import perm_test

################ DEFINITIONS #########################
lookup = {
    "Experts All": [0, 1, 2],
    "Novices All": [3, 4, 5],
    "Experts Resting": [0],
    "Experts Open Monitoring": [1],
    "Experts Compassion": [2],
    "Novices Resting": [3],
    "Novices Open Monitoring": [4],
    "Novices Compassion": [5],
    "Experts Meditating": [1, 2],
    "Novices Meditating": [4, 5],
    "Resting": [0, 3],
    "Compassion": [2, 5],
    "Open Monitoring": [1, 4],
    "Meditating": [1, 2, 4, 5],
}
TEST_LIST = []
# ## Intra (within) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: restricted, within subject
    ("Experts Resting", "Experts Compassion", "within"),
    ("Experts Resting", "Experts Open Monitoring", "within"),
    ("Experts Open Monitoring", "Experts Compassion", "within"),
    ("Experts Resting", "Experts Meditating", "within"),
    ("Novices Resting", "Novices Compassion", "within"),
    ("Novices Resting", "Novices Open Monitoring", "within"),
    ("Novices Open Monitoring", "Novices Compassion", "within"),
    ("Novices Resting", "Novices Meditating", "within"),
]
# ## Inter (between) Trait, Intra (within) State
TEST_LIST += [
    # Permutation: full
    ("Experts Resting", "Novices Resting", "full"),
    ("Experts Compassion", "Novices Compassion", "full"),
    ("Experts Open Monitoring", "Novices Open Monitoring", "full"),
]
# Permutation: restricted, across subject
TEST_LIST += [
    ("Experts Meditating", "Novices Meditating", "across"),
    ("Experts All", "Novices All", "across"),
]
## Inter (between) Trait, Inter (between) State
TEST_LIST += [
    # Permutation: free
    ("Experts Resting", "Novices Compassion", "full"),
    ("Experts Resting", "Novices Open Monitoring", "full"),
    ("Experts Compassion", "Novices Resting", "full"),
    ("Experts Compassion", "Novices Open Monitoring", "full"),
    ("Experts Open Monitoring", "Novices Resting", "full"),
    ("Experts Open Monitoring", "Novices Compassion", "full"),
]
# # Intra State (need to figure out these permutations)
TEST_LIST += [
    # Permutation: restricted, permute state
    ("Resting", "Compassion", "within"),
    ("Resting", "Open Monitoring", "within"),
    ("Compassion", "Open Monitoring", "within"),
    # Permutation: restricted, permute state (preserve # labels)
    ("Resting", "Meditating", "within"),
]
TEST_LIST = [(a, b) for a, b, c in TEST_LIST]

################ FUNCTIONS ###################
def k_sample_transform(inputs, ways=None):
    n_inputs = len(inputs)
    u = np.vstack(inputs)
    if np.var(u) == 0:
        raise ValueError("Test cannot be run, the inputs have 0 variance")

    if n_inputs == 2:
        n1 = inputs[0].shape[0]
        n2 = inputs[1].shape[0]
        v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
    else:
        if ways is None:
            ways = np.arange(n_inputs).reshape(-1, 1)
        vs = []
        input_lens = [len(input) for input in inputs]
        for way in np.array(ways).T:
            n_unique = len(np.unique(way))
            n_ways = len(way)
            v = np.zeros(shape=(n_ways, n_unique))
            v[np.arange(n_ways), way % n_unique] = 1
            vs.append(np.repeat(v, input_lens, axis=0))

        v = np.hstack(vs)

    return u, v


def get_files(
    path,
    level="(e|n)",
    subject="([0-9]{3})",
    task="(.+?)",
    filetype="h5",
):
    """
    Loads files from a directory

    Returns
    -------
    list of tuples, each (path, groups)
        groups is a tuple depending on which the inputs
    """
    files = []
    query = f"^{level}_sub-"
    query += f"{subject}_ses-1_"
    query += f"task-{task}.*\.{filetype}"
    for f in os.listdir(path):
        match = re.search(query, f)
        if match:
            files.append((f, match.groups()))

    return files


def get_latents(data_dir, n_components=None, ftype="h5"):
    tasks = ["restingstate", "openmonitoring", "compassion"]
    levels = ["e", "n"]

    gradients = []
    labels = []
    subj_ids = []

    for level in levels:
        for task in tasks:
            subgroup = []
            paths = get_files(data_dir, level=level, task=task, filetype=ftype)
            labels.append([level, task])
            n_load = len(paths)
            subjs = []

            for path, subj in paths:
                h5f = h5py.File(data_dir / path, "r")
                latent = h5f['latent'][:][..., :n_components]
                h5f.close()

                subgroup.append(latent)
                subjs.append(subj[0])

            gradients.append(subgroup)
            subj_ids.append(subjs)

    return gradients, labels, subj_ids


def get_k_sample_group(k_sample):
    if k_sample == "6":
        return [
            (
                "Experts Resting",
                "Experts Open Monitoring",
                "Experts Compassion",
                "Novices Resting",
                "Novices Open Monitoring",
                "Novices Compassion",
            )
        ]
    elif k_sample == "3N":
        return [
            (
                "Novices Resting",
                "Novices Open Monitoring",
                "Novices Compassion",
            )
        ]
    if k_sample == "3E":
        return [
            (
                "Experts Resting",
                "Experts Open Monitoring",
                "Experts Compassion",
            )
        ]
    else:
        raise ValueError(f"Undefined k_sample group label {k_sample}")


def discrim_test(X, Y, n_permutations, permute_groups):
    _, perm_blocks = np.unique(permute_groups, return_inverse=True)

    # Compute own distances if multiway or if zeroing in-group corrs
    X = pairwise_distances(X, metric="euclidean")
    Y = pairwise_distances(Y, metric="sqeuclidean")

    dcorr = Dcorr(compute_distance=None)
    stat, pvalue = dcorr.test(
        X,
        Y,
        reps=n_permutations,
        workers=-1,
        auto=False,
        perm_blocks=perm_blocks,
    )
    stat_dict = {"pvalue": pvalue, "test_stat": stat, "null_dist": dcorr.null_dist}

    return stat_dict


def compute_pvals(
    group_names,  # g1, g2,
    groups,
    labels,
    subjs,
    n_permutations,
    gradients,
    multiway=False,
):
    if len(group_names) == 2:
        name = f"{group_names[0]} vs. {group_names[1]}"
    else:
        name = f"{len(group_names)}-sample ({[lookup[g] for g in group_names]})"

    results_dict = {}

    subj_list = np.concatenate(
        [np.concatenate([np.asarray(subjs[i]) for i in lookup[g]]) for g in group_names]
    )

    print(name)
    if multiway:
        assert len(group_names) == 6, "multiway only available for 6-sample test"
        X, Y = k_sample_transform(
            [
                np.vstack([np.asarray(groups[i]) for i in lookup[g]])
                for g in group_names
            ],
            ways=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        )
    else:
        X, Y = k_sample_transform(
            [np.vstack([np.asarray(groups[i]) for i in lookup[g]]) for g in group_names]
        )

    for grads in gradients:
        Xg = X[:, :, grads]
        Xg = Xg.reshape(Xg.shape[0], -1)
        permute_groups = subj_list

        stat_dict = discrim_test(Xg, Y, n_permutations, permute_groups)
        results_dict[grads] = stat_dict

    return name, results_dict


def main(args):
    n_permutations = args.n_perms
    k_sample = args.k_sample
    multiway = args.multiway
    source_dir = Path(args.source)
    print(f"NEW RUN: DCORR, {n_permutations} permutations, k_sample={k_sample}")
    print(f"Loading data from directory: {source_dir}")

    groups, labels, subjs = get_latents(source_dir, n_components=3)

    ## Gradients
    gradients = [(0), (1), (2), (0, 1), (1, 2), (2, 0), (0, 1, 2)]
    data_dict = {}

    if k_sample is None:
        test_list = np.asarray(TEST_LIST)
        save_name = "2-sample"
    else:
        test_list = get_k_sample_group(k_sample)
        save_name = f"{k_sample}-sample"

    save_dir = Path("./dcorr_fmri_pvalues")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir / f"{save_name}_pvalues_{n_permutations}.csv", "w") as f:
        f.write(
            ",".join(["Comparison"] + [f'"Gradients {grads}"' for grads in gradients])
            + "\n"
        )
    for group_names in test_list:
        name, stat_dict = compute_pvals(
            group_names,
            groups=groups,
            labels=labels,
            subjs=subjs,
            n_permutations=n_permutations,
            gradients=gradients,
            multiway=multiway,
        )
        data_dict[name] = stat_dict

        print(f"Saving to {save_dir}")
        with open(save_dir / f"{save_name}_pvalues_{n_permutations}.csv", "a") as f:
            f.write(
                ",".join(
                    [f'"{name}"']
                    + [str(stat_dict[grads]["pvalue"]) for grads in gradients]
                )
                + "\n"
            )

        # with open(
        #     save_dir / f"{save_name}_results_dict_{n_permutations}.pkl", "wb"
        # ) as f:
        #     pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="", type=str, required=True)
    parser.add_argument("--n-perms", help="", type=int, default=1000)
    parser.add_argument(
        "--k-sample",
        help="Options {6: 6-sample All, 3N: 3-sample Novices, 3E: 3-sample Experts}",
        type=str,
        default=None,
    )
    parser.add_argument("--multiway", help="", action="store_true")
    args = parser.parse_args()

    main(args)
