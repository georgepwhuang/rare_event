import json

import numpy as np

from rare_event.angles import generate_thresh_angles
from rare_event.distributions import generate_pcoin_distribution, threshold_distribution, generate_dising_distribution ,generate_dising_entropy
from rare_event.qtvd import generate_qprobs

LAYERS = 4
K = 2.2
P = 0.1

original = generate_pcoin_distribution(P, LAYERS)
entropy = P * np.log2(P) + (1 - P) * np.log2(1 - P)

# original = generate_dising_distribution(1, 2, 3, 0.8, LAYERS)
# entropy = generate_dising_entropy(1, 2, 3, 0.8)

threshold = 2 ** (entropy * LAYERS * K / 2.0)
threshed, _, _ = threshold_distribution(original, threshold)

proj_set = generate_thresh_angles(600, 60, threshold)
qprobs = generate_qprobs(original, proj_set)

#with open("./data/importance_pcoin.json", "w") as f:
with open("./data/importance_dising.json", "w") as f:
    json.dump(
        {
            "layers": LAYERS,
            "threshold": float(threshold),
            "original": original.tolist(),
            "amplified": qprobs.tolist(),
            "thresholded": threshed.tolist(),
        },
        f,
    )