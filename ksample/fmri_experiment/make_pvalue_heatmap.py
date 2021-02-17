import numpy as np
import argparse
from pathlib import Path
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True, style="white", context="talk", font_scale=1)
PALETTE = sns.color_palette("Set1")

name_dict = {
    "Gradients 0": "Gradient 1",
    "Gradients 1": "Gradient 2",
    "Gradients 2": "Gradient 3",
    "Gradients (0, 1)": "Gradients 1,2",
    "Gradients (1, 2)": "Gradients 2,3",
    "Gradients (2, 0)": "Gradients 1,3",
    "Gradients (0, 1, 2)": "Gradients 1,2,3",
    "Gradients 0": "Gradient 1",
    "Gradients 1": "Gradient 2",
    "Gradients 2": "Gradient 3",
    "Gradients (0, 1)": "Gradients 1,2",
    "Gradients (1, 2)": "Gradients 2,3",
    "Gradients (2, 0)": "Gradients 1,3",
    "Gradients (0, 1, 2)": "Gradients 1,2,3",
    "Experts Resting vs. Experts Compassion": ["EXP", "res", "EXP", "com"],
    "Experts Resting vs. Experts Open Monitoring": ["EXP", "res", "EXP", "o m"],
    "Experts Open Monitoring vs. Experts Compassion": ["EXP", "o m", "EXP", "com"],
    "Experts Resting vs. Experts Meditating": ["EXP", "res", "EXP", "med"],
    "Novices Resting vs. Novices Compassion": ["NOV", "res", "NOV", "com"],
    "Novices Resting vs. Novices Open Monitoring": ["NOV", "res", "NOV", "o m"],
    "Novices Open Monitoring vs. Novices Compassion": ["NOV", "o m", "NOV", "com"],
    "Novices Resting vs. Novices Meditating": ["NOV", "res", "NOV", "med"],
    "Experts Resting vs. Novices Resting": ["EXP", "res", "NOV", "res"],
    "Experts Compassion vs. Novices Compassion": ["EXP", "com", "NOV", "com"],
    "Experts Open Monitoring vs. Novices Open Monitoring": ["EXP", "o m", "NOV", "o m"],
    "Experts Meditating vs. Novices Meditating": ["EXP", "med", "NOV", "med"],
    "Experts All vs. Novices All": ["EXP", "all", "NOV", "all"],
    "Experts Resting vs. Novices Compassion": ["EXP", "res", "NOV", "com"],
    "Experts Resting vs. Novices Open Monitoring": ["EXP", "res", "NOV", "o m"],
    "Experts Compassion vs. Novices Resting": ["EXP", "com", "NOV", "res"],
    "Experts Compassion vs. Novices Open Monitoring": ["EXP", "com", "NOV", "o m"],
    "Experts Open Monitoring vs. Novices Resting": ["EXP", "o m", "NOV", "res"],
    "Experts Open Monitoring vs. Novices Compassion": ["EXP", "o m", "NOV", "com"],
    "Resting vs. Compassion": ["ALL", "res", "ALL", "com"],
    "Resting vs. Open Monitoring": ["ALL", "res", "ALL", "o m"],
    "Compassion vs. Open Monitoring": ["ALL", "com", "ALL", "o m"],
    "Resting vs. Meditating": ["ALL", "res", "ALL", "med"],
}

label_dict = {
    "EXP": "EXP",
    "NOV": "NOV",
    "ALL": "ALL",
    "o m": "open",
    "med": "med ",
    "res": "rest",
    "com": "comp",
    "all": "all ",
}


def make_heatmap(source_dir, save_path):
    files = glob.glob(str(Path(source_dir) / "2-*.csv"))
    pvalues = pd.read_csv(files[0], index_col=0)

    pvalues.columns = [name_dict[v].split(" ")[-1] for v in pvalues.columns]

    index = [
        ["2", "X" if name_dict[v][0] == name_dict[v][2] else "", ""]
        + [label_dict[vv] for vv in name_dict[v]]
        for v in pvalues.index
    ]

    fmt_index = [
        "{:^1s} | {:^1s} | {:^1s} | {:^3s} {:<3s}, {:^3s} {:<3s}".format(*v)
        for v in index
    ]
    pvalues.index = fmt_index

    # Add pvalues from k-sample tests
    k_sample_paths = ["6-*.csv", "3E-*.csv", "3N-*.csv"]
    files = [glob.glob(str(Path(source_dir) / path))[0] for path in k_sample_paths]
    kpvals = np.vstack([pd.read_csv(f, index_col=0).values for f in files])

    # Scale
    kpvals = np.asarray(kpvals) * 7
    kpvals[1:, :] = kpvals[1:, :] * 2
    df = pd.DataFrame(kpvals, columns=pvalues.columns)
    df.index = [
        "6 | X | X | All states, traits",
        "3 | X |   | EXP states        ",
        "3 | X |   | NOV states        ",
    ]
    df[df > 1] = 1

    pvalues = pd.concat([df, pvalues])
    d = pvalues.values
    d[3:, :] *= np.multiply(*d[3:, :].shape)
    d[d > 1] = 1

    i_new = np.hstack(
        (
            pvalues.index[:3],
            [pvalues.index[15]],
            pvalues.index[3:15],
            pvalues.index[16:],
        )
    )
    d_new = np.vstack((d[:3], d[15], d[3:15], d[16:]))

    pvalues = pd.DataFrame(data=d_new, columns=pvalues.columns)
    pvalues.index = i_new

    mask = pvalues.copy()
    alpha = 0.05
    mask[:] = np.select([mask < alpha, mask >= alpha], ["X", ""], default=mask)

    f, ax = plt.subplots(1, figsize=(16, 9))
    ax = sns.heatmap(
        pvalues.transform("log10"),
        ax=ax,
        annot=mask,
        fmt="",
        square=False,
        linewidths=0.5,
        cbar_kws={"ticks": np.log10([0.01, 0.05, 0.1, 1])},
    )
    ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
    ax.collections[0].colorbar.set_ticklabels([0.01, 0.05, 0.1, 1])

    rot = 30

    # x labels
    loc, xlabels = plt.xticks()
    ax.xaxis.tick_top()
    ax.set_xticklabels(xlabels, rotation=0, ha="center")

    ax.set_yticklabels(
        ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"}
    )
    ax.text(
        -0.52, 1.02, "K", ha="left", va="bottom", rotation=rot, transform=ax.transAxes
    )
    ax.text(
        -0.46,
        1.02,
        "Multilevel",
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )
    ax.text(
        -0.39,
        1.02,
        "Multiway",
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )
    ax.text(
        -0.18,
        1.02,
        "Samples",
        ha="center",
        va="bottom",
        rotation=0,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        1.08,
        "Gradient(s)",
        ha="center",
        va="bottom",
        rotation=0,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", help="source directory with files", type=str, default=None
    )
    parser.add_argument(
        "-t", "--save", help="target file to save files", type=str, required=True
    )

    args = parser.parse_args()
    make_heatmap(args.source, args.save)
