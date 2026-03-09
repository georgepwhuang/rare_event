import json

import numpy as np
from matplotlib import pyplot as plt


def plot_importance(threshold, layers, original, amplified, thresholded):
    prob_thresh = threshold**2
    x = np.arange(0, 2**layers)
    y = np.concatenate(
        (
            np.arange(0, prob_thresh, round(prob_thresh / 3, 3))[:3],
            [prob_thresh],
            np.arange(0.2, 1, 0.2),
        )
    )

    figure_ratio = 0.2

    to_bin = lambda value: "{0:0{1}b}".format(value, layers)
    boost = (
        lambda value: figure_ratio * value / prob_thresh
        if value < prob_thresh
        else (1 - figure_ratio) * (value - prob_thresh) / (1 - prob_thresh)
        + figure_ratio
    )
    get_round = lambda value: round(value, 3)


    plt.rcParams.update({'font.size': 20, 'font.family': "serif","font.serif": "CMU Serif", 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
    plt.figure(figsize=(12, 6))
    plt.axhline(y=figure_ratio, color="gray", linestyle="--")
    plt.bar(x, thresholded, color="lightgrey", alpha=0.6, label="rare event", width=1)
    plt.bar(
        x - 0.1,
        list(map(boost, original)),
        color="C3",
        label="original",
        alpha=0.5,
        width=0.6,
        edgecolor="k",
    )
    plt.bar(
        x + 0.1,
        list(map(boost, amplified)),
        color="C0",
        label="amplified",
        alpha=0.5,
        width=0.6,
        edgecolor="k",
    )
    plt.ylim(0, 1)
    plt.xlabel("Event")
    plt.ylabel("Probability")
    plt.xticks(x, list(map(to_bin, x)), rotation=-45)
    plt.yticks(list(map(boost, y)), list(map(get_round, y)))
    plt.legend(loc="upper right")
    plt.tight_layout()
    #plt.savefig("./output/importance_pcoin.pdf", bbox_inches="tight")
    plt.savefig("./output/importance_dising.pdf", bbox_inches="tight")


#with open("./data/importance_pcoin.json", "r") as f:
with open("./data/importance_dising.json", "r") as f:
    data = json.load(f)

threshold = data["threshold"]
layers = data["layers"]
original = np.array(data["original"])
amplified = np.array(data["amplified"])
thresholded = np.array(data["thresholded"])

plot_importance(threshold, layers, original, amplified, thresholded)