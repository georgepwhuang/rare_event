import sys
from rare_event import iMPS, MarkovianRecurrentQuantumCircuit, RealDiagonalBlockEncoding, QET, plot_uniform, plot_importance
import numpy as np
from scipy.special import erf
import pennylane as qml
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries

import warnings
warnings.filterwarnings("ignore")


LAYERS = 4
POLYDEG = 100
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

delta = 50

threshold = np.sqrt(2**((p*np.log2(p)+(1-p)*np.log2(1-p))*LAYERS)/10.0)

func = lambda x: (erf(delta*(x + threshold)) - erf(delta*(x - threshold))) / 2
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: np.where(np.abs(x) < threshold, 1, 0)


poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=POLYDEG,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*POLYDEG)

(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True,
    signal_operator="Wx")

response.PlotQSPResponse(
    phiset,
    pcoefs=poly,
    target=true_func,
    sym_qsp=True,
    simul_error_plot=True)

def convert_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = np.pi / 4 - (num_angles % 4) * np.pi/2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = np.pi / 4

    return angles + update_vals

def complement(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = 3 * np.pi / 4 + (num_angles % 4) * np.pi/2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = - np.pi / 4

    return -angles + update_vals

phiset = convert_angles(phiset)

dev = qml.device("lightning.qubit", wires=base_qubits)

@qml.qnode(dev)
def circuit():
    MarkovianRecurrentQuantumCircuit(wires=list(range(base_qubits)),
                                     memory_state_prep_list=[mem_0, mem_1], 
                                     initial_state=0, 
                                     transition=transition)
    return qml.probs(list(range(1, base_qubits)))

if SIMULATE:
    dev2 = qml.device("lightning.qubit", wires = base_qubits + 3)
    all_wires = list(range(base_qubits + 2))
    wires = list(range(2, base_qubits + 2))
    main_wires = list(range(3, base_qubits + 2))
    other_wires = [0, 1, 2]
    ancilla_wires = [1]
    control_wires = [1, 2]
    rotation_wire = 0
    work_wire = base_qubits + 2
else: 
    dev2 = qml.device("lightning.qubit", wires = 2 * base_qubits + 4)
    all_wires = list(range(2 * base_qubits + 3))
    wires = list(range(base_qubits + 3, 2 * base_qubits + 3))
    main_wires = list(range(base_qubits + 4, 2 * base_qubits + 3))
    ancilla_wires = list(range(1, base_qubits + 3))
    control_wires = list(range(1, base_qubits + 4))
    rotation_wire = 0
    work_wire = 2 * base_qubits + 3

@qml.prod
def reflect(wires):
    qml.X(wires[-1])
    qml.ctrl(qml.Z(wires=wires[-1]), 
             control=wires[:-1], 
             control_values=[0] * (len(wires) - 1))
    qml.X(wires[-1])

@qml.prod
def uniformStatePrep():
    for wire in main_wires:
        qml.Hadamard(wires=wire)
    QET(RealDiagonalBlockEncoding,
        U=MarkovianRecurrentQuantumCircuit, 
        wires=wires, 
        ancilla_wires=ancilla_wires, 
        control_wires=control_wires, 
        rotation_wire=rotation_wire, 
        angles=phiset, 
        simulate=SIMULATE, 
        memory_state_prep_list=[mem_0, mem_1], 
        initial_state=0, 
        transition=transition)

@qml.prod
def importanceStatePrep():
    MarkovianRecurrentQuantumCircuit(wires=wires,
                                     memory_state_prep_list=[mem_0, mem_1], 
                                     initial_state=0, 
                                     transition=transition)
    QET(RealDiagonalBlockEncoding,
        U=MarkovianRecurrentQuantumCircuit, 
        wires=wires, 
        ancilla_wires=ancilla_wires, 
        control_wires=control_wires, 
        rotation_wire=rotation_wire, 
        angles=phiset, 
        simulate=SIMULATE, 
        memory_state_prep_list=[mem_0, mem_1], 
        initial_state=0, 
        transition=transition)
    
U = importanceStatePrep()
O = reflect(other_wires)

@qml.qnode(dev2)
def circuitSim():
    importanceStatePrep()
    qml.AmplitudeAmplification(U, O, iters=55, fixed_point=True, work_wire=work_wire, p_min=0.99)
    return qml.probs(main_wires)

original = circuit()
thresholded = true_func(pow(original, 0.5))
amplified = circuitSim()
print(amplified)
plot_importance(threshold, LAYERS, original, amplified, thresholded)

try:
    fig, ax = qml.draw_mpl(circuitSim)()
    fig.savefig('algo.png')
except ValueError:
    print("Figure too long", file=sys.stderr)