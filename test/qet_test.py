import numpy as np
from rare_event import iMPS, MarkovianRecurrentQuantumCircuit, RealDiagonalBlockEncoding, QET
import pennylane as qml

def pcoin(p):
    """
    transition matrix for pcoin

    -------------------
    Parameters
    -------------------
    p:float
    -------------------
    
    """
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

angles = [-0.20409113, -0.91173829, 0.91173829, 0.20409113]

dev = qml.device("default.qubit", wires=base_qubits)
dev2 = qml.device("default.qubit", wires= 2*base_qubits+3)
dev3 = qml.device("default.qubit", wires= base_qubits+1)

@qml.qnode(dev)
def circuit():
    MarkovianRecurrentQuantumCircuit(wires=list(range(base_qubits)),memory_state_prep_list=[mem_0, mem_1], initial_state=0, transition=transition)
    return qml.state()

@qml.qnode(dev2)
def circuitGate():
    QET(RealDiagonalBlockEncoding,U=MarkovianRecurrentQuantumCircuit, wires=list(range(base_qubits+3, 2*base_qubits+3)), ancilla_wires=list(range(1, base_qubits+3)), control_wires=list(range(1, base_qubits+4)), rotation_wire =[0], simulate=False, angles=angles,memory_state_prep_list=[mem_0, mem_1], initial_state=0, transition=transition)
    return qml.state()

@qml.qnode(dev3)
def circuitSimu():
    QET(RealDiagonalBlockEncoding,U=MarkovianRecurrentQuantumCircuit, wires=list(range(1, base_qubits+1)), ancilla_wires=[0], control_wires=[0, 1], angles=angles,memory_state_prep_list=[mem_0, mem_1], initial_state=0, transition=transition)
    return qml.state()

x = circuit().real
print((5*x**3-3*x)/2)
mat = qml.matrix(circuitGate)()
print(np.round(np.diag((mat))[:2**(base_qubits-1)], 5).real)

mat = qml.matrix(circuitSimu)()
print(np.round(np.diag((mat))[:2**(base_qubits-1)], 5).real)

fig, ax = qml.draw_mpl(circuitGate)()
fig.savefig('qet.png')

fig, ax = qml.draw_mpl(circuitSimu)()
fig.savefig('qetsimu.png')