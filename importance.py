import numpy as np
from matplotlib import pyplot as plt

from rare_event.angles import generate_thresh_angles
from rare_event.qtvd import generate_qprobs
from rare_event import tranmatrix as tm

LAYERS = 4
K = 2.5

def plot_importance(threshold, layers, original, amplified, thresholded):
    prob_thresh = threshold**2
    x = np.arange(0, 2**layers)
    y = np.concatenate((np.arange(0, prob_thresh, round(prob_thresh/3, 3))[:3], [prob_thresh], np.arange(0.2, 1, 0.2)))
    
    figure_ratio = 0.2
    
    to_bin = lambda x: "{0:0{1}b}".format(x, layers)
    boost = lambda x: figure_ratio * x / prob_thresh if x < prob_thresh else (1-figure_ratio)*(x-prob_thresh) / (1-prob_thresh) + figure_ratio
    get_round = lambda x: round(x, 3)

    def addlabels(x, original, amplified):
        for i in range(len(x)):
            plt.text(i, boost(amplified[i])+1e-2, f"x{round(amplified[i]/original[i], 1)}", ha = 'center')
    
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.family': "serif"})
    plt.figure(figsize=(12,6))
    plt.axhline(y=figure_ratio, color='gray', linestyle='--')
    plt.bar(x, thresholded, color='lightgrey', alpha=0.6, label='rare event', width=1)
    plt.bar(x-0.1, list(map(boost, original)), color='C3', label='original', alpha=0.5, width=0.6, edgecolor='k')
    plt.bar(x+0.1, list(map(boost, amplified)), color='C0', label='amplified', alpha=0.5, width=0.6, edgecolor='k')
    plt.ylim(0, 1)
    #addlabels(x, original, amplified)
    plt.xlabel("Event")
    plt.ylabel("Probability")
    plt.xticks(x, list(map(to_bin, x)), rotation=-45)
    plt.yticks(list(map(boost, y)), list(map(get_round,y)))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('importance_pcoin.pdf',bbox_inches='tight')

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

def generate_dising_distribution(J0, delta, chi, T, L, start=0):
    trans_mat = tm.dising(J0, delta, chi, T)
    trans_mat = trans_mat[0] + trans_mat[1]
    original = []
    for n in range(pow(2, L)):
        res = "{0:0{1}b}".format(start, chi) + "{0:0{1}b}".format(n, L)
        prob = 1
        for i in range(L):
            a = int(''.join(map(str, res[0+i:chi+i])), 2)
            b = int(''.join(map(str, res[1+i:chi+1+i])), 2)
            prob *= trans_mat[b][a]
        original.append(prob)
    original = np.array(original)
    return original

def generate_dising_entropy(J0, delta, chi, T):
    trans_mat = tm.dising(J0, delta, chi, T)
    trans_mat = trans_mat[0] + trans_mat[1]
    trans_mat = np.array(trans_mat)
    entropy = np.sum(trans_mat * np.nan_to_num(np.log2(trans_mat), neginf=0), axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
    for i in range(len(eigenvalues)):
        if np.allclose(eigenvalues[i], 1):
            stationary = eigenvectors[:,i]
            stationary = stationary/np.sum(stationary)
            return np.real(np.sum(stationary * entropy))
    raise RuntimeError("Stationary distribution not found.")

original = generate_pcoin_distribution(0.1, LAYERS)
entropy = 0.1*np.log2(0.1)+0.9*np.log2(0.9)

threshold = 2**(entropy*LAYERS*K/2.0)
threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
rare_num = np.sum(threshed)
actual = original * threshed
p_rare = np.sum(actual)
actual = actual / np.sum(actual)

# Generate quantum 
proj_set = generate_thresh_angles(600, 60, threshold)
qprobs = generate_qprobs(original, proj_set)

plot_importance(threshold, LAYERS, original, qprobs, threshed)