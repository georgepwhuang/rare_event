import numpy as np
import matplotlib.pyplot as plt

def draw(threshold, layers, original, amplified, thresholded):
    prob_thresh = threshold**2
    x = np.arange(0, 2**layers)
    y = np.concatenate((np.arange(0, prob_thresh+1e-12, prob_thresh/5), np.arange(0, 1, 0.2)))

    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i, boost(y[i])+1e-2, round(y[i],3), ha = 'center')

    figure_ratio = 0.2

    to_bin = lambda x: "{0:0{1}b}".format(x, layers)
    boost = lambda x: figure_ratio * x / prob_thresh if x < prob_thresh else (1-figure_ratio)*(x-prob_thresh) / (1-prob_thresh) + figure_ratio
    get_round = lambda x: round(x, 3)

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,6))
    plt.axhline(y=figure_ratio, color='gray', linestyle='--')
    plt.bar(x, thresholded, color='lightgrey', label='rare event', width=1)
    plt.bar(x-0.1, list(map(boost, original)), color='red', label='original', alpha=0.6, width=0.6, edgecolor='k')
    plt.bar(x+0.1, list(map(boost, amplified)), color='blue', label='amplified', alpha=0.5, width=0.6, edgecolor='k')
    plt.ylim(0, 1)
    addlabels(x, amplified)
    plt.xlabel("Event")
    plt.ylabel("Probability")
    plt.xticks(x, list(map(to_bin, x)))
    plt.yticks(list(map(boost, y)), list(map(get_round,y)))
    plt.legend()

    plt.suptitle("Rare Event Identification")
    plt.tight_layout()
    plt.savefig('results.png')