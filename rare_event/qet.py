import pennylane as qml

def ProjCtrlPhaseShift(control_wires, target_wire, phi):
    qml.MultiControlledX(control_wires=control_wires, wires= target_wire, control_values=[0]*len(control_wires))
    qml.RZ(phi = 2*phi, wires=target_wire)
    qml.MultiControlledX(control_wires=control_wires, wires= target_wire, control_values=[0]*len(control_wires))
    
def QET(BE, wires, ancilla_wires, angles, simulate=True, control_wires=None, rotation_wire=None, *args, **kwargs):
    deg = len(angles)
    if control_wires is None:
        control_wires = ancilla_wires
    if simulate:
        assert rotation_wire is None
        uncontrolled_wires = list(set(wires + ancilla_wires) - set(control_wires))
        dim = 2**len(uncontrolled_wires)
        for i in range(0, deg-2, 2):
            qml.PCPhase(dim=dim, wires=control_wires+uncontrolled_wires, phi = angles[-i-1])
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            qml.PCPhase(dim=dim, wires=control_wires+uncontrolled_wires, phi = angles[-i-2])
            qml.adjoint(BE)(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
        qml.PCPhase(dim=dim, wires=control_wires+uncontrolled_wires, phi = angles[-i-3])
        if len(angles) % 2 == 0:
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            qml.PCPhase(dim=dim, wires=control_wires+uncontrolled_wires, phi = angles[0])
    else:
        for i in range(0, deg-2, 2):
            ProjCtrlPhaseShift(control_wires=control_wires, target_wire=rotation_wire, phi = angles[-i-1])
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            ProjCtrlPhaseShift(control_wires=control_wires, target_wire=rotation_wire, phi = angles[-i-2])
            qml.adjoint(BE)(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
        ProjCtrlPhaseShift(control_wires=control_wires, target_wire=rotation_wire, phi = angles[-i-3])
        if len(angles) % 2 == 0:
            BE(wires=wires, ancilla_wires=ancilla_wires, simulate=simulate, *args, **kwargs)
            ProjCtrlPhaseShift(control_wires=control_wires, target_wire=rotation_wire, phi = angles[0])