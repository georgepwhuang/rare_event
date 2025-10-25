import json
from matplotlib import pyplot as plt
import numpy as np

with open('dising.json', 'r') as f:
    data = json.load(f)
x = np.array(data["x"])
q = np.array(data["q"])
c = np.array(data["c"])
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': "serif"})
plt.figure(figsize=(12,6))
plt.plot(x, q, label='Quantum')
plt.plot(x, c, label='Classical', linestyle='dashed')
plt.xticks(np.arange(0, np.max(x), 500))
plt.xlabel("Queries")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig('dising.pdf',bbox_inches='tight')