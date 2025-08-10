import json
from matplotlib import pyplot as plt
import numpy as np

with open('pcoin.json', 'r') as f:
    data = json.load(f)
x = np.array(data["x"])
q = np.array(data["q"])
c = np.array(data["c"])
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,6))
plt.plot(x, q, color='red', label='Quantum')
plt.plot(x, c, color='blue', label='Classical')
plt.xticks(x[4::5])
plt.xlabel("Degree")
plt.ylabel("TVD")
plt.legend()
plt.title("Simulations of a Dyson Ising Chain")
plt.tight_layout()
plt.savefig('pcoin.pdf')