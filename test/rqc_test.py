import numpy as np
from rare_event import iMPS, RecurrentQuantumCircuit, RealDiagonalBlockEncoding
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
dev = qml.device("default.qubit", wires=base_qubits)
dev2 = qml.device("default.qubit", wires=2 + 2*base_qubits)
dev3 = qml.device("qiskit.aer", wires=1 + base_qubits)

@qml.qnode(dev)
def circuit():
    RecurrentQuantumCircuit(wires=list(range(base_qubits)),memory_state_prep=mem_0, transition=transition)
    return qml.state()

@qml.qnode(dev2)
def circuitGate():
    RealDiagonalBlockEncoding(RecurrentQuantumCircuit, wires=list(range(base_qubits+2, 2*base_qubits+2)), ancilla_wires=list(range(base_qubits+2)), simulate=False, memory_state_prep=mem_0, transition=transition)
    return qml.state()

@qml.qnode(dev3)
def circuitSimu():
    RealDiagonalBlockEncoding(RecurrentQuantumCircuit, wires=list(range(1, base_qubits+1)), ancilla_wires=[0], memory_state_prep=mem_0, transition=transition)
    return qml.state()

print(circuit().real)
mat = qml.matrix(circuitGate)()
print(np.round(np.diag((mat))[:2**base_qubits], 8).real)

mat = qml.matrix(circuitSimu)()
print(np.round(np.diag((mat))[:2**base_qubits], 8).real)

fig, ax = qml.draw_mpl(circuitGate)()
fig.savefig('rqc.png')

fig, ax = qml.draw_mpl(circuitSimu)()
fig.savefig('rqcsimu.png')