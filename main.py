from rare_event import iMPS, RecurrentQuantumCircuit, RealDiagonalBlockEncoding, QET
import numpy as np
from scipy.special import erf
import pennylane as qml
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries

import warnings
warnings.filterwarnings("ignore")

def pcoin(p):
    T0 = np.array([
        [1.-p,p],
        [0.,0.]
    ])
    T1 = np.array([
        [0.,0.],
        [p,1.-p]
    ])
    return np.array([T0,T1])

p = 0.1
imps_pcoin = iMPS.from_tmatrix(pcoin(p))
mem_0 = imps_pcoin.lft_cform().en_past([0])
mem_1 = imps_pcoin.lft_cform().en_past([1])
transition = imps_pcoin.to_unitary()

layers = 2
base_qubits = layers + 1

delta = 10
threshold = 0.2

simu = True

func = lambda x: (erf(delta*(x + threshold)) - erf(delta*(x - threshold))) / 2
polydeg = 6
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: np.where(np.abs(x) < threshold, 1, 0) * max_scale

poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    cheb_samples=2*polydeg)

phiset = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='laurent',
    signal_operator="Wx")

#response.PlotQSPResponse(
#    phiset,
#    pcoefs=poly.coef,
#    target=true_func)

def convert_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 4

    return angles + update_vals

phiset = convert_angles(phiset)

dev = qml.device("lightning.qubit", wires=base_qubits)
if simu:
    dev2 = qml.device("lightning.qubit", wires=2*base_qubits+2)
else: 
    dev2 = qml.device("lightning.qubit", wires=2*base_qubits+3)

@qml.qnode(dev)
def circuit():
    RecurrentQuantumCircuit(wires=list(range(base_qubits)),memory_state_prep=mem_0, transition=transition)
    return qml.state()

if simu:
    wires = list(range(base_qubits+2, 2*base_qubits+2))
    ancilla_wires = list(range(base_qubits+2))
else: 
    wires = list(range(base_qubits+3, 2*base_qubits+3))
    ancilla_wires = list(range(base_qubits+3))
    
@qml.qnode(dev2)
def circuit2():
    QET(RealDiagonalBlockEncoding,U=RecurrentQuantumCircuit, wires=wires, ancilla_wires=ancilla_wires, angles=phiset, simulate=simu, memory_state_prep=mem_0, transition=transition)
    return qml.state()

x = circuit().real
print(func(x))

mat = qml.matrix(circuit2)()
print(np.diag((mat * (np.abs(mat)>1e-8)).real[:2**base_qubits,:2**base_qubits]))