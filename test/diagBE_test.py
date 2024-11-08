import pennylane as qml
from rare_event import RealDiagonalBlockEncoding
import numpy as np

dev = qml.device("default.qubit", wires=2)
dev2 = qml.device("default.qubit", wires=6)
dev3 = qml.device("default.qubit", wires=3)

def TestLayer(wires, phi):
    qml.X(wires[0])
    qml.CRY(phi=phi, wires=wires)
    
@qml.qnode(dev)
def circuit():
    TestLayer(wires=[0, 1], phi = 0.5)
    return qml.state()

@qml.qnode(dev2)
def circuitGate():
    RealDiagonalBlockEncoding(TestLayer, wires=[4, 5], ancilla_wires=[0, 1, 2, 3], simulate=False, phi=0.5)
    return qml.state()

@qml.qnode(dev3)
def circuitSimu():
    RealDiagonalBlockEncoding(TestLayer, wires=[1, 2], ancilla_wires=[0], phi=0.5)
    return qml.state()


print(circuit().real)
mat = qml.matrix(circuitGate)()
print(np.round(np.diag((mat))[:4], 5).real)

mat = qml.matrix(circuitSimu)()
print(np.round(np.diag((mat))[:4], 5).real)

fig, ax = qml.draw_mpl(circuitGate)()
fig.savefig('diagBE.png')