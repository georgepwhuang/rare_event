import numpy as np
import math
import rare_event.tranmatrix as tm
from rare_event.angles import generate_thresh_angles
from rare_event.amplify import generate_qtvd, generate_ctvd
import json

LAYERS = 6
K = 2.5

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

qres = []
cres = []
cquery = []
qsum = []

gap = 10
grid = np.arange(10, 401, gap)

for deg in grid:
    #Generate probability distribution
    original = generate_dising_distribution(1, 2, 3, 0.8, LAYERS)
    entropy = generate_dising_entropy(1, 2, 3, 0.8)
    delta = math.ceil(deg/5)

    threshold = 2**(entropy*LAYERS*K/2.0)
    threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
    actual = original * threshed
    actual = actual / np.sum(actual)

    proj_set = generate_thresh_angles(deg, delta, threshold)
    qtvd, qprob_sum = generate_qtvd(original, actual, proj_set)
    queries = int(math.ceil(deg / np.sqrt(qprob_sum)))
    ctvd = generate_ctvd(original, actual, queries, threshold, repeat=1000, groups=10)
    qres.append(qtvd)
    cres.append(ctvd)
    cquery.append(queries)
    qsum.append(qprob_sum)

with open("dising.json", "w") as f:
    json.dump({"x":grid.tolist(), "q":qres, "c":cres, "cquery":cquery, "qsum":qsum}, f)