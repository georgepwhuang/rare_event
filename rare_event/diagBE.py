import pennylane as qml
from pennylane.typing import TensorLike


def MultiControlledZ(wires, control_values=None):
    if control_values is None:
        control_values = [0] * (len(wires) - 1)
    qml.Hadamard(wires=[wires[-1]])
    qml.MultiControlledX(wires=wires, control_values=control_values)
    qml.Hadamard(wires=[wires[-1]])


def R(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.PauliX(wires=wires[0])
    MultiControlledZ(wires=wires[1:n+1]+[wires[0]])
    qml.PauliX(wires=wires[0])


def hatZ(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.PauliZ(wires=wires[n])


def hatH(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.Hadamard(wires=wires[n])


def hatS(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.S(wires=wires[n])


def Uc(base, wires, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    if isinstance(base, TensorLike):
        qml.ControlledQubitUnitary(base, control_wires=wires[n], wires=wires[:n], control_values=[0], unitary_check=True)
    elif isinstance(base, qml.operation.Operator) or callable(base):
        qml.ctrl(base, control=wires[n], control_values=[0])(wires=wires[:n], *args, **kwargs)
    

def C(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    for i in range(n):
        qml.Toffoli(wires=[wires[n], wires[n+i+1], wires[i]])


def W(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    hatH(wires)
    Uc(base, wires, *args, **kwargs)
    C(wires)
    if int(bool(p)) == 1:
        hatS(wires)
    hatH(wires)
    

def G(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    hatZ(wires)
    qml.adjoint(W)(base, wires, p, *args, **kwargs)
    R(wires)
    W(base, wires, p, *args, **kwargs)


def RealDiagonalBlockEncoding(U, wires, ancilla_wires, p=0, *args, **kwargs):
    assert len(ancilla_wires) == len(wires) + 2
    qml.Hadamard(wires=ancilla_wires[0])
    W(base=U, wires=ancilla_wires[1:]+wires, p=p, *args, **kwargs)
    qml.ctrl(G, control=ancilla_wires[0], control_values=[0])(base=U, wires=ancilla_wires[1:]+wires,p=p, **kwargs)
    qml.ctrl(qml.adjoint(G), control=ancilla_wires[0], control_values=[1])(base=U, wires=ancilla_wires[1:]+wires,p=p, *args, **kwargs)
    qml.Hadamard(wires=ancilla_wires[0])
    qml.adjoint(W)(base=U, wires=ancilla_wires[1:]+wires, p=p, *args, **kwargs)
    qml.PauliX(wires=ancilla_wires[0])
    qml.PauliZ(wires=ancilla_wires[0])
    qml.PauliX(wires=ancilla_wires[0])
    