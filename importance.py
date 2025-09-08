import numpy as np
import math
from matplotlib import pyplot as plt

from rare_event.angles import generate_thresh_angles
from rare_event.qtvd import generate_qprobs

LAYERS = 4
K = 2.2
p = 0.1
deg = 50

def plot_importance(threshold, layers, original, amplified, thresholded):
    prob_thresh = threshold**2
    x = np.arange(0, 2**layers)
    y = np.concatenate((np.arange(0, prob_thresh+1e-12, prob_thresh/5), np.arange(0.2, 1, 0.2)))
    
    figure_ratio = 0.2
    
    to_bin = lambda x: "{0:0{1}b}".format(x, layers)
    boost = lambda x: figure_ratio * x / prob_thresh if x < prob_thresh else (1-figure_ratio)*(x-prob_thresh) / (1-prob_thresh) + figure_ratio
    get_round = lambda x: round(x, 3)

    def addlabels(x, original, amplified):
        for i in range(len(x)):
            plt.text(i, boost(amplified[i])+1e-2, f"x{round(amplified[i]/original[i], 2)}", ha = 'center')
    
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12,6))
    plt.axhline(y=figure_ratio, color='gray', linestyle='--')
    plt.bar(x, thresholded, color='lightgrey', label='rare event', width=1)
    plt.bar(x-0.1, list(map(boost, original)), color='red', label='original', alpha=0.6, width=0.6, edgecolor='k')
    plt.bar(x+0.1, list(map(boost, amplified)), color='blue', label='amplified', alpha=0.5, width=0.6, edgecolor='k')
    plt.ylim(0, 1)
    addlabels(x, original, amplified)
    plt.xlabel("Event")
    plt.ylabel("Probability")
    plt.xticks(x, list(map(to_bin, x)))
    plt.yticks(list(map(boost, y)), list(map(get_round,y)))
    plt.legend()
    plt.tight_layout()
    plt.savefig('importance.pdf')

def generate_pcoin_distribution(p, L):
    original = []
    for n in range(pow(2, L)):
        mask = (1 << L) - 1
        bitFlips = (n & mask) ^ (n >> 1)
        flipCount = bin(bitFlips).count('1')
        prob = pow(p, flipCount)*pow(1-p, L-flipCount)
        original.append(prob)
    original = np.array(original)
    return original

original = generate_pcoin_distribution(p, LAYERS)
entropy = (p*np.log2(p)+(1-p)*np.log2(1-p))

threshold = 2**(entropy*LAYERS*K/2.0)
threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
rare_num = np.sum(threshed)
actual = original * threshed
p_rare = np.sum(actual)
actual = actual / np.sum(actual)

# Generate quantum 
proj_set = generate_thresh_angles(400, 40, threshold)
qprobs = generate_qprobs(original, proj_set)

plot_importance(threshold, LAYERS, original, qprobs, threshed)