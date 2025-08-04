import numpy as np
from scipy.special import erf
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries
from matplotlib import pyplot as plt
import math
import tranmatrix as tm
import sys

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

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m)

def convert_qsvt_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = - np.pi / 4 * (2 * num_angles - 1)
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = np.pi / 4

    return angles + update_vals

def get_fixed_point_angles(iters, p_min):
    delta = np.sqrt(1 - p_min)
    gamma = np.cosh(np.arccosh(1 / delta) / iters) ** -1

    alphas = [
        float(2 * np.arctan(1 / (np.tan(2 * np.pi * j / iters) * np.sqrt(1 - gamma**2))))
        for j in range(1, iters // 2 + 1)
    ]
    betas = [-alphas[-j] for j in range(1, iters // 2 + 1)]
    return alphas[: iters // 2], betas[: iters // 2]

qres = []
cres = []
qcount = []

gap = 50
grid = np.arange(50, 601, gap)

for deg in grid:
    #Generate probability distribution
    original = generate_dising_distribution(1, 2, 2, 0.8, LAYERS)
    entropy = generate_dising_entropy(1, 2, 2, 0.8)
    delta = math.ceil(deg/6)

    queries = deg + 1

    threshold = 2**(entropy*LAYERS*K/2.0)
    threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
    actual = original * threshed
    actual = actual / np.sum(actual)

    def convert_qsvt_angles(angles):
        num_angles = len(angles)
        update_vals = np.zeros(num_angles)

        update_vals[0] = - np.pi / 4 * (2 * num_angles - 1)
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = np.pi / 4

        return angles + update_vals
    
    #Generate projector angles
    func = lambda x: (erf(delta * (x + threshold)) - erf(delta * (x - threshold))) / 2

    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=deg,
        max_scale=0.99,
        chebyshev_basis=True,
        cheb_samples=2*deg)
    
    pcoefs = poly.coef
    # force odd coefficients to be zero, since the polynomial must be even
    pcoefs[1::2] = 0

    (proj_set, _, _) = angle_sequence.QuantumSignalProcessingPhases(
        pcoefs,
        method='sym_qsp',
        chebyshev_basis=True,
        signal_operator="Wx")
    

    proj_set = convert_qsvt_angles(proj_set)

    #Produce threshold block encoding
    statevector = np.sqrt(original)
    state_diag = np.diag(statevector)
    inv_diag = np.diag(np.sqrt(1 - original))
    block_encoding = np.vstack((np.hstack((state_diag, inv_diag)), 
                                np.hstack((inv_diag, -state_diag))))
    length = original.size
    qet_be_pos = np.diag(np.hstack((np.repeat(np.exp(1j * proj_set[-1]), length), 
                                    np.repeat(np.exp(-1j * proj_set[-1]), length))))
    qet_be_neg = np.diag(np.hstack((np.repeat(np.exp(-1j * proj_set[-1]), length), 
                                    np.repeat(np.exp(1j * proj_set[-1]), length))))
    for idx, phi in enumerate(reversed(proj_set[:-1])):
        if idx % 2 == 0:
            qet_be_pos = block_encoding @ qet_be_pos
            qet_be_neg = block_encoding @ qet_be_neg
        else:
            qet_be_pos = block_encoding.T @ qet_be_pos
            qet_be_neg = block_encoding.T @ qet_be_neg
        phase_pos = np.diag(np.hstack((np.repeat(np.exp(1j * phi), length), 
                                       np.repeat(np.exp(-1j * phi), length))))
        phase_neg = np.diag(np.hstack((np.repeat(np.exp(-1j * phi), length), 
                                       np.repeat(np.exp(1j * phi), length))))
        qet_be_pos = phase_pos @ qet_be_pos
        qet_be_neg = phase_neg @ qet_be_neg
    qet_be_zero = 0.5 * (qet_be_pos + qet_be_neg)
    qet_be_one = 0.5 * (qet_be_pos - qet_be_neg)
    qet_be = np.vstack((np.hstack((qet_be_zero, qet_be_one)), 
                        np.hstack((qet_be_one, qet_be_zero))))
    
    proj_statevector = qet_be @ np.hstack((statevector, np.zeros(3*length)))
    probs = np.abs(proj_statevector)**2
    probs = probs[:length]
    prob_sum = np.sum(probs)
    probs = probs / prob_sum

    qtvd = np.linalg.norm(actual - probs, ord=1)/2.0

    UNI_QUERIES = int(math.ceil(queries / np.sqrt(prob_sum)))

    qcount.append(UNI_QUERIES)

    ctvd = []

    for i in range(100):
    #Classical approach
        samples = np.random.choice(len(original), UNI_QUERIES, p=original)
        values, counts = np.unique(samples, return_counts=True)
        values = values.astype(np.int32)
        full_indices = np.arange(len(original))
        uni_samp = np.zeros_like(full_indices)
        uni_samp[values] = counts
        uni_samp = uni_samp / np.sum(uni_samp)
        mask = np.where(pow(uni_samp, 0.5) < threshold, 1, 0)

        imp_samp = original * mask 
        imp_samp = imp_samp / np.sum(imp_samp)
        ctvd.append(np.linalg.norm(actual - imp_samp, ord=1)/2.0)
    ctvd = np.stack(ctvd)
    ctvd = np.mean(ctvd, axis=0)

    qres.append(qtvd)
    cres.append(ctvd)

#Plot
x = grid
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,6))
plt.plot(x, qres, color='red', label='Quantum')
plt.plot(x, cres, color='blue', label='Classical')
plt.xticks(x)
plt.xlabel("Degree")
plt.ylabel("TVD")
plt.legend()
plt.title("Simulations of a Dyson-Ising chain")
plt.tight_layout()
plt.savefig('dising.pdf')
