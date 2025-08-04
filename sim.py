import numpy as np
from scipy.special import erf
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries, PolySign
from matplotlib import pyplot as plt

LAYERS = 1
POLYDEG = 100
AA_ITER = 99
K = 1.5
SIMULATE = True

QUERIES = (POLYDEG + 1) * AA_ITER

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m)

def Zget_fixed_point_angles(iters, p_min):
    delta = np.sqrt(1 - p_min)
    gamma = np.cos(np.arccos(1 / delta, dtype=np.complex128) / iters, dtype=np.complex128) ** -1

    alphas = [
        float(2 * np.arctan(1 / (np.tan(2 * np.pi * j / iters) * np.sqrt(1 - gamma**2))))
        for j in range(1, iters // 2 + 1)
    ]
    betas = [-alphas[-j] for j in range(1, iters // 2 + 1)]
    return alphas[: iters // 2], betas[: iters // 2]


UNI_QUERIES = int(POLYDEG * AA_ITER /2)
IMP_QUERIES = int(AA_ITER /2)

qres = []
cres = []

for p in np.arange(0.1, 1, 0.1)[:1]:

    #Generate probability distribution
    original = []
    for n in range(pow(2, LAYERS)):
        mask = (1 << LAYERS) - 1
        bitFlips = (n & mask) ^ (n >> 1)
        flipCount = bin(bitFlips).count('1')
        prob = pow(p, flipCount)*pow(1-p, LAYERS-flipCount)
        original.append(prob)
    original = np.array(original)
    threshold = 2**((p*np.log2(p)+(1-p)*np.log2(1-p))*LAYERS*K/2.0)
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
    
    delta = 40
    #Generate projector angles
    func = lambda x: (erf(delta * (x + threshold)) - erf(delta * (x - threshold))) / 2

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
    
    # response.PlotQSPResponse(
    # proj_set,
    # pcoefs=poly,
    # target=func,
    # sym_qsp=True,
    # simul_error_plot=True)

    proj_set = convert_qsvt_angles(proj_set)

    #Generate fixed-point amplitude amplfication angles

    poly, _ = PolySign().generate(
        degree=AA_ITER,
        delta=30,
        max_scale=0.999,
        return_scale=True,
        ensure_bounded=True,
        chebyshev_basis=True,
        cheb_samples=2 * AA_ITER)
    
    (aa_set, _, _) = angle_sequence.QuantumSignalProcessingPhases(
        poly,
        method='sym_qsp',
        chebyshev_basis=True,
        signal_operator="Wx")
    
    # response.PlotQSPResponse(
    # aa_set,
    # pcoefs=poly,
    # target=np.sign,
    # sym_qsp=True,
    # simul_error_plot=True)

    # aa_set = convert_qsvt_angles(aa_set)

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
    print(qet_be[:2, :2])

    #Produce AA block encoding
    extended_statevector = np.hstack((np.ones(1), np.zeros(4 * length -1)))
    state_proj = np.outer(extended_statevector, extended_statevector)
    inv_proj = np.eye(4 * length) - state_proj
    aa_be_pos = np.exp(1j * aa_set[-1]) * state_proj + np.exp(-1j * aa_set[-1]) * inv_proj
    aa_be_neg = np.exp(-1j * aa_set[-1]) * state_proj + np.exp(1j * aa_set[-1]) * inv_proj
    for idx, phi in enumerate(reversed(aa_set[:-1])):
        if idx % 2 == 0:
            aa_be_pos = qet_be @ aa_be_pos
            aa_be_neg = qet_be @ aa_be_neg
            phase_pos = np.diag(np.hstack((np.repeat(np.exp(1j * phi), length), 
                                           np.repeat(np.exp(-1j * phi), 3*length))))
            phase_neg = np.diag(np.hstack((np.repeat(np.exp(-1j * phi), length), 
                                           np.repeat(np.exp(1j * phi), 3*length))))
            aa_be_pos = phase_pos @ aa_be_pos
            aa_be_neg = phase_neg @ aa_be_neg
        else: 
            aa_be_pos = qet_be.T @ aa_be_pos
            aa_be_neg = qet_be.T @ aa_be_neg
            phase_pos = np.exp(1j * phi) * state_proj + np.exp(-1j * phi) * inv_proj
            phase_neg = np.exp(-1j * phi) * state_proj + np.exp(1j * phi) * inv_proj
            aa_be_pos = phase_pos @ aa_be_pos
            aa_be_neg = phase_neg @ aa_be_neg
    aa_be_zero = 0.5 * (aa_be_pos + aa_be_neg)
    aa_be_one = 0.5 * (aa_be_pos - aa_be_neg)
    aa_be = np.vstack((np.hstack((aa_be_zero, aa_be_one)), 
                       np.hstack((aa_be_one, aa_be_zero))))

    final = np.hstack((statevector, np.zeros(7 * length)))
    final = aa_be @ final
    amplified = final[:length]**2

    qtvd = np.linalg.norm(actual - amplified, ord=1)/2.0

    #Classical approach
    samples = np.random.choice(len(original), UNI_QUERIES, p=original)
    values, counts = np.unique(samples, return_counts=True)
    values = values.astype(np.int32)
    full_indices = np.arange(len(original))
    uni_samp = np.zeros_like(full_indices)
    uni_samp[values] = counts
    uni_samp = uni_samp / np.sum(uni_samp)
    mask = np.where(pow(uni_samp, 0.5) < threshold, 1, 0)
    values, counts = np.unique(samples, return_counts=True)
    values = values.astype(np.int32)
    imp_samp = np.zeros_like(full_indices)
    imp_samp[values] = counts
    imp_samp = imp_samp * mask 
    imp_samp = imp_samp / np.sum(imp_samp)
    ctvd = np.linalg.norm(actual - imp_samp, ord=1)/2.0

    qres.append(qtvd)
    cres.append(ctvd)

#Plot
x = np.arange(0.1, 1, 0.1)
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,6))
plt.bar(x-0.04, qres, 0.04)
plt.bar(x, cres, 0.04)
plt.xlabel("Event")
plt.ylabel("TVD")
plt.tight_layout()
plt.savefig('pcoin.png')
