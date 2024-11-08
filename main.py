import sys
from rare_event import iMPS, MarkovianRecurrentQuantumCircuit, RealDiagonalBlockEncoding, QET
import numpy as np
from scipy.special import erf
import pennylane as qml
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries

import warnings
warnings.filterwarnings("ignore")


LAYERS = 3

DELTA = 20
THRESHOLD = 0.1

SIMULATE = True

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

base_qubits = LAYERS + 1

func = lambda x: (erf(DELTA*(x + THRESHOLD)) - erf(DELTA*(x - THRESHOLD))) / 2
polydeg = 10
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: np.where(np.abs(x) < THRESHOLD, 1, 0)


poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True,
    signal_operator="Wx")

#response.PlotQSPResponse(
#    phiset,
#    pcoefs=poly,
#    target=true_func,
#    sym_qsp=True,
#    simul_error_plot=True)

def convert_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = np.pi / 2 - (3 + len(angles) % 4) * np.pi / 2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 2

    return angles + update_vals

phiset = convert_angles(phiset)

dev = qml.device("lightning.qubit", wires=base_qubits)

@qml.qnode(dev)
def circuit():
    MarkovianRecurrentQuantumCircuit(wires=list(range(base_qubits)),memory_state_prep_list=[mem_0, mem_1], initial_state=0, transition=transition)
    return qml.state()

if SIMULATE:
    dev2 = qml.device("default.qubit", wires=base_qubits+1)
    wires = list(range(1, base_qubits+1))
    ancilla_wires = [0]
    control_wires = [0, 1]
    rotation_wire = None
else: 
    dev2 = qml.device("default.qubit", wires=2*base_qubits+3)
    wires = list(range(base_qubits+3, 2*base_qubits+3))
    ancilla_wires = list(range(1, base_qubits+3))
    control_wires = list(range(1, base_qubits+4))
    rotation_wire = [0]
    
@qml.qnode(dev2)
def circuitGate():
    QET(RealDiagonalBlockEncoding,U=MarkovianRecurrentQuantumCircuit, wires=wires, ancilla_wires=ancilla_wires, control_wires=control_wires, rotation_wire=rotation_wire, angles=phiset, simulate=SIMULATE, memory_state_prep_list=[mem_0, mem_1], initial_state=0, transition=transition)
    return qml.state()

x = circuit().real[0:2**(base_qubits-1)]
print(x)
print(true_func(x))

mat = qml.matrix(circuitGate)()
print(np.round(np.diag((mat))[:2**(base_qubits-1)], 5).real)

try:
    fig, ax = qml.draw_mpl(circuitGate, decimals=2)()
    fig.savefig('algo.png')
except:
    print("Figure too long", file=sys.stderr)