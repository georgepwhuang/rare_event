import numpy as np

def generate_qtvd(prob, actual, angles):
    statevector = np.sqrt(prob)
    state_diag = np.diag(statevector)
    inv_diag = np.diag(np.sqrt(1 - prob))
    block_encoding = np.vstack((np.hstack((state_diag, inv_diag)), 
                                np.hstack((inv_diag, -state_diag))))
    length = prob.size
    qet_be_pos = np.diag(np.hstack((np.repeat(np.exp(1j * angles[-1]), length), 
                                    np.repeat(np.exp(-1j * angles[-1]), length))))
    qet_be_neg = np.diag(np.hstack((np.repeat(np.exp(-1j * angles[-1]), length), 
                                    np.repeat(np.exp(1j * angles[-1]), length))))
    for idx, phi in enumerate(reversed(angles[:-1])):
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
    qprobs = np.abs(proj_statevector)**2
    qprobs = qprobs[:length]
    qprob_sum = np.sum(qprobs)
    qprobs = qprobs / qprob_sum
    
    qtvd = np.linalg.norm(actual - qprobs, ord=1)/2.0
    return qtvd
