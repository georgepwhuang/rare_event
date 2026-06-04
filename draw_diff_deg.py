import json
from matplotlib import pyplot as plt
import numpy as np

with open('./data/deg_diff_pcoin.json', 'r') as f:
#with open('./data/deg_diff_dising.json', 'r') as f:
    data = json.load(f)
x = np.array(data["x"])
q = np.array(data["q"])
c = np.array(data["c"])
plt.rcParams.update({'font.size': 20, 'font.family': "serif","font.serif": "CMU Serif", 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
plt.figure(figsize=(12,6))
plt.plot(x, q, label='Quantum')
plt.plot(x, c, label='Classical', linestyle='dashed')
plt.xticks(np.arange(0, np.max(x), 500))
plt.xlabel("Applications of (Quantum) Sampler")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
#plt.savefig('./output/deg_diff_pcoin.pdf',bbox_inches='tight')
plt.savefig('./output/deg_diff_dising.pdf',bbox_inches='tight')