import numpy as np
from rare_event import iMPS, RecurrentQuantumCircuit, RealDiagonalBlockEncoding, QET
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

@qml.qnode(dev)
def circuit():
    RecurrentQuantumCircuit(wires=list(range(base_qubits)),memory_state_prep=mem_0, transition=transition)
    return qml.state()

@qml.qnode(dev2)
def circuitGate():
    QET(RealDiagonalBlockEncoding,U=RecurrentQuantumCircuit, wires=list(range(base_qubits+3, 2*base_qubits+3)), ancilla_wires=list(range(base_qubits+3)), angles=angles,memory_state_prep=mem_0, transition=transition)
    return qml.state()

x = circuit().real
print((5*x**3-3*x)/2)
mat = qml.matrix(circuitGate)()
print(np.diag((mat * (np.abs(mat)>0.0001)).real[:2**base_qubits,:2**base_qubits]))

fig, ax = qml.draw_mpl(circuitGate)()
fig.savefig('qet.png')