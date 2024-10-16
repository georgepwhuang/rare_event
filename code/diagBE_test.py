import pennylane as qml
from diagBE import RealDiagonalBlockEncoding
import numpy as np

dev = qml.device("default.qubit", wires=2)
dev2 = qml.device("default.qubit", wires=6)

def TestLayer(wires, phi):
    qml.X(wires[0])
    qml.CRY(phi=phi, wires=wires)
    
@qml.qnode(dev)
def circuit():
    TestLayer(wires=[0, 1], phi = 0.5)
    return qml.state()

@qml.qnode(dev2)
def circuitGate():
    RealDiagonalBlockEncoding(TestLayer, wires=[4, 5], ancilla_wires=[0, 1, 2, 3], phi=0.5)
    return qml.state()


print(circuit().real)
mat = qml.matrix(circuitGate)()
print(np.diag((mat * (np.abs(mat)>0.0001)).real[:4,:4]))

fig, ax = qml.draw_mpl(circuitGate)()
fig.savefig('diagBE.png')