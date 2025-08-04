import numpy as np
from scipy.special import erf
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries
from matplotlib import pyplot as plt
import math

LAYERS = 6
POLYDEG = 200
DELTA = 60
K = 1.5

QUERIES = POLYDEG + 1

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

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m)

def convert_qsvt_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = - np.pi / 4 * (2 * num_angles - 1)
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = np.pi / 4

    return angles + update_vals

qres = []
cres = []

gap = 0.02
grid = np.arange(0.1, 0.3, gap)
for p in grid:
    #Generate probability distribution
    original = generate_pcoin_distribution(p, LAYERS)

    threshold = 2**((p*np.log2(p)+(1-p)*np.log2(1-p))*LAYERS*K/2.0)
    threshed = np.where(pow(original, 0.5) < threshold, 1, 0)
    actual = original * threshed
    actual = actual / np.sum(actual)
    
    #Generate projector angles
    func = lambda x: (erf(DELTA * (x + threshold)) - erf(DELTA * (x - threshold))) / 2

    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=POLYDEG,
        max_scale=0.99,
        chebyshev_basis=True,
        cheb_samples=2*POLYDEG)
    
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
    
    proj_statevector = qet_be @ np.hstack((statevector, np.zeros(3 * length)))
    probs = np.abs(proj_statevector)**2
    probs = probs[:length]
    prob_sum = np.sum(probs)
    probs = probs/prob_sum

    qtvd = np.linalg.norm(actual - probs, ord=1)/2.0

    UNI_QUERIES = int(math.ceil(QUERIES / np.sqrt(prob_sum)))

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
plt.bar(x-gap/5.0, qres, gap/2.5, color='red', label='Quantum', alpha=0.6, edgecolor='k')
plt.bar(x+gap/5.0, cres, gap/2.5, color='blue', label='Classical', alpha=0.5, edgecolor='k')
plt.xticks(x)
plt.legend()
plt.xlabel("Event Probability")
plt.ylabel("TVD")
plt.title("Simulations of a p-coin")
plt.legend()
plt.tight_layout()
plt.savefig('pcoin.pdf')
