from statistics import NormalDist

import numpy as np

def generate_ctvd(probs, actual, queries, threshold, repeat=1000):
    ctvd = []
    for _ in range(repeat):
        samples = np.random.choice(len(probs), queries, p=probs)
        values, counts = np.unique(samples, return_counts=True)
        values = values.astype(np.int32)
        uni_samp = np.zeros(len(probs))
        uni_samp[values] = counts
        uni_samp = uni_samp / queries
        mask = np.where(uni_samp < threshold**2, 1, 0)

        imp_samp = probs * mask
        acceptance_mass = np.sum(imp_samp)
        imp_samp = imp_samp / acceptance_mass
        ctvd.append(np.linalg.norm(actual - imp_samp, ord=1)/2.0)
    ctvd = np.mean(ctvd, axis=0)
    return ctvd


def generate_ctvd_confidence(
    probs, actual, queries, threshold, repeat=1000,
    z=NormalDist().inv_cdf(0.975),
):
    if repeat < 2 or not np.isfinite(z) or z <= 0:
        raise ValueError("repeat must be at least 2 and z must be positive and finite.")
    trials = np.array([
        generate_ctvd(probs, actual, queries, threshold, repeat=1)
        for _ in range(repeat)
    ])
    if not np.all(np.isfinite(trials)):
        raise RuntimeError(f"Non-finite classical TVD at query budget {queries}.")
    mean = float(np.mean(trials))
    std = float(np.std(trials, ddof=1))
    sem = float(std / np.sqrt(repeat))
    return mean, std, sem, max(0.0, mean-z*sem), min(1.0, mean+z*sem)
