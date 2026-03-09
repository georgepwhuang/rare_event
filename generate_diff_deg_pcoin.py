import numpy as np
import math
from rare_event.angles import generate_thresh_angles
from rare_event.qtvd import generate_qtvd
from rare_event.ctvd import generate_ctvd
from rare_event.distributions import generate_pcoin_distribution, threshold_distribution
import json

LAYERS = 8
K = 2.2
p = 0.1

x = []
qres = []
cres = []

gap = 20
grid = np.arange(20, 850, gap)

original = generate_pcoin_distribution(p, LAYERS)
entropy = (p*np.log2(p)+(1-p)*np.log2(1-p))

threshold = 2**(entropy*LAYERS*K/2.0)
threshed, actual, p_rare = threshold_distribution(original, threshold)

for deg in grid:
    # Generate probability distribution
    queries = int(deg * math.ceil(1 / np.sqrt(p_rare)))
    delta = int(deg/10)

    # Generate quantum 
    proj_set = generate_thresh_angles(deg, delta, threshold)
    qtvd = generate_qtvd(original, actual, proj_set)
    qres.append(qtvd)

    # Generate classical 
    ctvd = generate_ctvd(original, actual, queries, threshold, repeat=1000)
    cres.append(ctvd)
    x.append(queries)

with open("./data/deg_diff_pcoin.json", "w") as f:
    json.dump({"x":x, 
               "q":qres, 
               "c":cres}, f)