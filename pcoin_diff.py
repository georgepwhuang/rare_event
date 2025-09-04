import numpy as np
from rare_event.angles import generate_thresh_angles
from rare_event.qtvd import generate_qtvd, generate_ctvd
import math
import json

LAYERS = 6
deg = 200
delta = 60
K = 1.5

queries = delta + 1

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

qres = []
cres = []

gap = 0.02
grid = np.arange(0.1, 0.3, gap)
for p in grid:
    #Generate probability distribution
    original = generate_pcoin_distribution(p, LAYERS)
    entropy = (p*np.log2(p)+(1-p)*np.log2(1-p))
    threshold = 2**(entropy*LAYERS*K/2.0)
    threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
    actual = original * threshed
    actual = actual / np.sum(actual)
    
    #Generate projector angles
    proj_set = generate_thresh_angles(deg, delta, threshold)
    qtvd, qprob_sum = generate_qtvd(original, actual, proj_set)
    queries = int(math.ceil((deg + 1) / np.sqrt(qprob_sum)))
    ctvd, ctvd_std = generate_ctvd(original, actual, queries, repeat=100)
    qres.append(qtvd)
    cres.append(ctvd)
    
with open("pcoin.json", "w") as f:
    json.dump({"x":grid, "q":qres, "c":cres}, f)