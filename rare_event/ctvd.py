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
        mask_size = np.sum(mask)

        imp_samp = probs * mask 
        imp_samp = imp_samp / np.sum(imp_samp)
        ctvd.append(np.linalg.norm(actual - imp_samp, ord=1)/2.0)
    cvtd = np.mean(ctvd, axis=0)
    return cvtd