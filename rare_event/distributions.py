import numpy as np

from rare_event import tranmatrix as tm


def generate_pcoin_distribution(p, layers):
    original = []
    for n in range(pow(2, layers)):
        mask = (1 << layers) - 1
        bit_flips = (n & mask) ^ (n >> 1)
        flip_count = bin(bit_flips).count("1")
        prob = pow(p, flip_count) * pow(1 - p, layers - flip_count)
        original.append(prob)
    return np.array(original)


def generate_dising_distribution(J0, delta, chi, T, layers, start=0):
    trans_mat = tm.dising(J0, delta, chi, T)
    trans_mat = trans_mat[0] + trans_mat[1]
    original = []
    for n in range(pow(2, layers)):
        res = "{0:0{1}b}".format(start, chi) + "{0:0{1}b}".format(n, layers)
        prob = 1
        for i in range(layers):
            a = int("".join(map(str, res[0 + i : chi + i])), 2)
            b = int("".join(map(str, res[1 + i : chi + 1 + i])), 2)
            prob *= trans_mat[b][a]
        original.append(prob)
    return np.array(original)


def generate_dising_entropy(J0, delta, chi, T):
    trans_mat = tm.dising(J0, delta, chi, T)
    trans_mat = trans_mat[0] + trans_mat[1]
    trans_mat = np.array(trans_mat)
    entropy = np.sum(trans_mat * np.nan_to_num(np.log2(trans_mat), neginf=0), axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
    for i in range(len(eigenvalues)):
        if np.allclose(eigenvalues[i], 1):
            stationary = eigenvectors[:, i]
            stationary = stationary / np.sum(stationary)
            return np.real(np.sum(stationary * entropy))
    raise RuntimeError("Stationary distribution not found.")


def threshold_distribution(original, threshold):
    threshed = np.where(np.sqrt(original) < threshold, 1, 0)
    actual = original * threshed
    p_rare = np.sum(actual)
    actual = actual / np.sum(actual)
    return threshed, actual, p_rare