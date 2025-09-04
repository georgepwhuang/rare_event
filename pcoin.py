import numpy as np
import math
from rare_event.angles import generate_thresh_angles
from rare_event.qtvd import generate_qtvd
from rare_event.ctvd import generate_ctvd
import json

LAYERS = 7
K = 1.5
p = 0.2
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
cquery = []
qsum = []

gap = 10
grid = np.arange(10, 1001, gap)

original = generate_pcoin_distribution(p, LAYERS)
entropy = (p*np.log2(p)+(1-p)*np.log2(1-p))

threshold = 2**(entropy*LAYERS*K/2.0)
threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
rare_num = np.sum(threshed)
actual = original * threshed
p_rare = np.sum(actual)
actual = actual / np.sum(actual)

for queries in grid:
    #Generate probability distribution
    deg = int(math.ceil(queries / np.sqrt(p_rare)))
    delta = math.ceil(deg/5)

    #Generate quantum 
    proj_set = generate_thresh_angles(deg, delta, threshold)
    qtvd = generate_qtvd(original, actual, proj_set)
    qres.append(qtvd)

    #Generate classical 
    ctvd = generate_ctvd(original, actual, queries, threshold, repeat=1000)
    cres.append(ctvd)

with open("pcoin.json", "w") as f:
    json.dump({"x":grid.tolist(), 
               #"q":qres, 
               "c":cres}, f)