import pennylane as qml

def ProjCtrlPhaseShift(control_wires, target_wire, phi):
    qml.MultiControlledX(control_wires=control_wires, wires= target_wire, control_values=[0]*len(control_wires))
    qml.RZ(phi = 2*phi, wires=target_wire)
    qml.MultiControlledX(control_wires=control_wires, wires= target_wire, control_values=[0]*len(control_wires))
    
def QET(BE, wires, ancilla_wires, angles, simulate=True, *args, **kwargs):
    deg = len(angles)
    if simulate:
        dim = 2**len(wires)
        for i in range(0, deg-2, 2):
            qml.PCPhase(dim=dim, wires=ancilla_wires+wires, phi = angles[-i-1])
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            qml.PCPhase(dim=dim, wires=ancilla_wires+wires, phi = angles[-i-2])
            qml.adjoint(BE)(wires=wires, ancilla_wires=ancilla_wires, *args, **kwargs)
        qml.PCPhase(dim=dim, wires=ancilla_wires+wires, phi = angles[-i-3])
        if len(angles) % 2 == 0:
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            qml.PCPhase(dim=dim, wires=ancilla_wires+wires, phi = angles[0])
    else:
        for i in range(0, deg-2, 2):
            ProjCtrlPhaseShift(control_wires=ancilla_wires[1:], target_wire=ancilla_wires[0], phi = angles[-i-1])
            BE(wires=wires, ancilla_wires=ancilla_wires[1:], simulate=simulate, *args, **kwargs)
            ProjCtrlPhaseShift(control_wires=ancilla_wires[1:], target_wire=ancilla_wires[0], phi = angles[-i-2])
            qml.adjoint(BE)(wires=wires, ancilla_wires=ancilla_wires[1:], simulate=simulate, *args, **kwargs)
        ProjCtrlPhaseShift(control_wires=ancilla_wires[1:], target_wire=ancilla_wires[0], phi = angles[-i-3])
        if len(angles) % 2 == 0:
            BE(wires=wires, ancilla_wires=ancilla_wires[1:], simulate=simulate, *args, **kwargs)
            ProjCtrlPhaseShift(control_wires=ancilla_wires[1:], target_wire=ancilla_wires[0], phi = angles[0])