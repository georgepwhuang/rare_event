import pennylane as qml
from pennylane.typing import TensorLike
import numpy as np


def MultiControlledZ(wires, control_values=None):
    if control_values is None:
        control_values = [0] * (len(wires) - 1)
    qml.ctrl(qml.Z(wires=wires[-1]), 
             control=wires[:-1], 
             control_values=control_values)

def R(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.PauliX(wires=wires[0])
    MultiControlledZ(wires=wires[1:n+1]+[wires[0]])
    qml.PauliX(wires=wires[0])

def Uc(base, wires, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    if isinstance(base, TensorLike):
        qml.ControlledQubitUnitary(base, 
                                   control_wires=wires[n], 
                                   wires=wires[:n], 
                                   control_values=[0], 
                                   unitary_check=True)
    elif isinstance(base, qml.operation.Operator) or callable(base):
        qml.ctrl(base, control=wires[n], 
                 control_values=[0])(wires=wires[:n], *args, **kwargs)
        
def Uc_adj(base, wires, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    if isinstance(base, TensorLike):
        qml.adjoint(qml.ControlledQubitUnitary)(base, 
                                                control_wires=wires[n], 
                                                wires=wires[:n], 
                                                control_values=[0], 
                                                unitary_check=True)
    elif isinstance(base, qml.operation.Operator) or callable(base):
        qml.ctrl(qml.adjoint(base), 
                 control=wires[n], 
                 control_values=[0])(wires=wires[:n], *args, **kwargs)

def C(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    for i in range(n):
        qml.Toffoli(wires=[wires[n], wires[n+i+1], wires[i]])
        
def C_adj(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    for i in range(n-1, -1, -1):
        qml.Toffoli(wires=[wires[n], wires[n+i+1], wires[i]])

def W(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.Hadamard(wires[n])
    Uc(base, wires, *args, **kwargs)
    C(wires)
    if bool(p):
        qml.S(wires[n])
    qml.Hadamard(wires[n])
    
def W_adj(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.Hadamard(wires[n])
    if bool(p):
        qml.adjoint(qml.S)(wires[n])
    C_adj(wires)
    Uc_adj(base, wires, *args, **kwargs)
    qml.Hadamard(wires[n])

def G(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qml.PauliZ(wires[n])
    W_adj(base, wires, p, *args, **kwargs)
    R(wires)
    W(base, wires, p, *args, **kwargs)
    
def G_adj(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    W_adj(base, wires, p, *args, **kwargs)
    R(wires)
    W(base, wires, p, *args, **kwargs)
    qml.PauliZ(wires[n])

def RealDiagonalBlockEncoding(U, wires, ancilla_wires,
                              p=0, simulate=True, *args, **kwargs):
    if simulate:
        assert len(ancilla_wires) == 1
        statevector = get_statevector(U, wires, *args, **kwargs)
        if bool(p):
            qml.BlockEncode(np.diag(statevector.imag), 
                            wires=ancilla_wires+wires)
        else: 
            qml.BlockEncode(np.diag(statevector.real), 
                            wires=ancilla_wires+wires)
    else:
        assert len(ancilla_wires) == len(wires) + 2
        qml.Hadamard(wires=ancilla_wires[0])
        W(base=U, 
          wires=ancilla_wires[1:]+wires, 
          p=p, *args, **kwargs)
        qml.ctrl(G, control=ancilla_wires[0], 
                 control_values=[0])(base=U, wires=ancilla_wires[1:]+wires,
                                     p=p, **kwargs)
        qml.ctrl(G_adj, control=ancilla_wires[0], 
                 control_values=[1])(base=U, wires=ancilla_wires[1:]+wires,
                                     p=p, *args, **kwargs)
        qml.Hadamard(wires=ancilla_wires[0])
        W_adj(base=U, wires=ancilla_wires[1:]+wires, p=p, *args, **kwargs)
        qml.PauliX(wires=ancilla_wires[0])
        qml.PauliZ(wires=ancilla_wires[0])
        qml.PauliX(wires=ancilla_wires[0])
        
def ShiftedDiagonalBlockEncoding(U, wires, ancilla_wires,
                                 p=0, simulate=True, identity_ratio=1, *args, **kwargs):
    lcu_wire = ancilla_wires[-1]
    theta = np.arctan(np.sqrt(np.abs(identity_ratio)))*2
    phi = np.angle(identity_ratio)
    qml.RY(-theta, lcu_wire)
    qml.ctrl(RealDiagonalBlockEncoding, 
                  control=lcu_wire, 
                  control_values=0)(U, wires, ancilla_wires[:-1],
                                      p, simulate, *args, **kwargs)
    qml.PhaseShift(phi, lcu_wire)
    qml.RY(theta, lcu_wire)
    
@qml.QueuingManager.stop_recording()
def get_statevector(U, wires, *args, **kwargs):
    dev = qml.device("lightning.qubit", wires=len(wires))
    @qml.qnode(dev)
    def circuit():
        U(wires=list(range(len(wires))), *args, **kwargs)
        return qml.state()
    return circuit()